# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from transformers import AutoTokenizer

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
            
            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
            
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "response_mask", "response_attention_mask", "format_reward", "num_turns"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        #uid = data.non_tensor_batch["uid"]
        #response_mask_all = data.batch["response_mask"]
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        reward_lst = []
        true_means_lst = []
        turn_starts_lst = []
        next_line_prob_lst = []
       # from transformers import AutoTokenizer
       # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
        for mini_iter, micro_batch in enumerate(micro_batches):
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            response_attention_mask = model_inputs["response_attention_mask"]
            response_mask = model_inputs["response_mask"]
            response_ids = model_inputs["responses"]
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
                prob = torch.exp(log_probs)
                #after_last_mask = (response_mask.flip(-1).cumsum(-1) == 0).flip(-1)  # only attend to the last turn of gt text
                gt_mask = response_attention_mask * (torch.ones_like(response_mask) - response_mask)               
                padded_mask = F.pad(gt_mask, (1, 0), "constant", 0)
                # A turn starts where the mask changes from 0 to 1
                turn_starts = (padded_mask[:, 1:] - padded_mask[:, :-1] == 1).float()

                # 2. Create a unique ID for each turn in each batch sample
                turn_ids = torch.cumsum(turn_starts, dim=-1)
                masked_turn_ids = (turn_ids * gt_mask).long()

                # 3. Use scatter_add_ for segmented sum to get per-turn sums and counts
                batch_size = prob.shape[0]
                max_turns = masked_turn_ids.max().item() if masked_turn_ids.numel() > 0 else 0
                
                masked_log_probs = prob * gt_mask

                turn_sums = torch.zeros(batch_size, max_turns + 1, device=log_probs.device, dtype=log_probs.dtype)
                turn_counts = torch.zeros(batch_size, max_turns + 1, device=log_probs.device, dtype=gt_mask.dtype)
               # """
                batch_size, N = masked_turn_ids.shape
                tid1_list = [] 
                tid2_list = []
                #res_0_list = []
                res_1_list = []
                res_2_list = []
                count1 = 0
                response_length_mine = model_inputs["responses"].size(-1)
                for n in range(N):
                    tid = masked_turn_ids[0, n].item()
                  #  if tid == 2:
                  #      res_0_list.append(response_ids[0, n].item())
                    if tid == 1:
                    #    if count1 == 0 or count1 == 1 or count1 == 2 or count1 == 3:
                    #        print("symbolinfirst", tokenizer.decode([response_ids[0, n].item()]), "endfirst")
                        tid1_list.append((masked_log_probs[0, n]).item())
                        res_1_list.append(response_ids[0, n].item())
                        count1 = n + 1
                    if tid == 6:
                   #     if count2 == 0 :
                   #         print("symbolinsix", response_ids[0, n].item(), "endsix")
                        tid2_list.append((masked_log_probs[0, n]).item())
                        res_2_list.append(response_ids[0, n].item())
                
                #if (mini_iter == 0 or mini_iter == 1):
                  #  """
                 #   if not calculate_entropy:
                 #       print("tid1: ", tid1_list, "endtid1")
                 #       print("tid6: ", tid2_list, "endtid6")

                turn_sums.scatter_add_(1, masked_turn_ids, masked_log_probs)
                turn_counts.scatter_add_(1, masked_turn_ids, gt_mask)

                format_reward = model_inputs["format_reward"]
             #   print("deeeebug", turn_means.size(), turn_means)
             #   print("format_reward", format_reward.shape, format_reward)
                format_reward = format_reward[:, :max_turns].to(turn_starts.device)
                format_reward = F.pad(format_reward, (1, 0), 'constant', 0)

                turn_means_noformat = turn_sums / turn_counts.clamp_min(1)

                turn_means = turn_means_noformat + format_reward - torch.ones_like(format_reward)
                format_mask = (format_reward > 0.5)            # dtype: bool, same shape as format
                turn_means = turn_means.masked_fill(~format_mask, 0.0)

            #    total_return = torch.sum(turn_means[:, 1:], dim=-1) / model_inputs["num_turns"]  # newly added

             #  """
                window = 3
                x = turn_means.unsqueeze(1)               # → [B,1,L]
                x_padded = F.pad(x, (0, window-1))  # → [B,1,L + window-1]
                kernel = torch.ones(1, 1, window, dtype=x.dtype, device=x.device)
                #  turn_means = F.conv1d(x_padded, kernel).squeeze(1)      # → [B,1,(L+window-1)−window+1] = [B,L]
              ### mask_ones = torch.ones_like(x)
                mask_ones = (torch.arange(x.size(-1), device=x.device)[None, :] <= model_inputs["num_turns"][:, None]).to(x.dtype).unsqueeze(1)
                mask_ones_padded = F.pad(mask_ones, (0, window - 1))
                sum_window = F.conv1d(x_padded, kernel)          # sums over available elements
                cnt_window = F.conv1d(mask_ones_padded, kernel)          # counts of real elements (no zeros)

                turn_means = (sum_window / cnt_window.clamp_min(1)).squeeze(1)  # [B, L]
                turn_means[:, 0] = 0.0
              #  """
                # 5. Scatter the means back to a sequence-shaped tensor
                # First, map the mean of a turn to every token in that turn
                per_token_means = torch.gather(turn_means, 1, masked_turn_ids)
                # Then, create the sparse reward tensor by only keeping values at turn starts
                reward_scores = (per_token_means * turn_starts).detach()
            ##    reward_scores = (total_return[:, None] * turn_starts).detach()

            num_turns = model_inputs["num_turns"]
            true_means_lst.append(torch.sum(turn_means, dim=-1) / num_turns)# nonzero_counts)

            log_probs_lst.append(log_probs)
            reward_lst.append(reward_scores)
            turn_starts_lst.append(turn_starts)
            if calculate_entropy:
                entropy_lst.append(entropy)
        true_means_ = torch.concat(true_means_lst, dim=0)
        reward_scores = torch.concat(reward_lst, dim=0)
        log_probs = torch.concat(log_probs_lst, dim=0)
        turn_starts_ = torch.concat(turn_starts_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            reward_scores = restore_dynamic_batch(reward_scores, batch_idx_list)
            true_means_ = restore_dynamic_batch(true_means_, batch_idx_list)
            turn_starts_ = restore_dynamic_batch(turn_starts_, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
        
        return log_probs, entropys, reward_scores, true_means_, turn_starts_ #, advantages

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "response_attention_mask",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                         #   "critic/rewards/mean": torch.mean(reward_scores).detach().item(),
                         #   "critic/rewards/max": torch.max(reward_scores).detach().item(),
                         #   "critic/rewards/min": torch.min(reward_scores).detach().item(),
                         #   "critic/advantages/mean": torch.mean(advantages).detach().item(),
                         #   "critic/advantages/max": torch.max(advantages).detach().item(),
                         #   "critic/advantages/min": torch.min(advantages).detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_ntp(self, data: DataProto):
        self.actor_module.train()
        temperature = 1.0

        select_keys = ["responses","response_mask","input_ids","attention_mask","position_ids","response_attention_mask"]
        has_mmi = "multi_modal_inputs" in data.non_tensor_batch.keys()
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=(["multi_modal_inputs"] if has_mmi else []))

        mini_batches = data.split(self.config.ntp_mini_batch_size)
        metrics = {}

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ntp_mini_batch_size // self.config.ntp_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ntp_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                # (optional) exact token-mean across micro-batches
                if getattr(self.config, "ntp_exact_token_mean", False):
                    total_tokens = 0
                    for mb in micro_batches:
                        total_tokens += mb.batch["response_attention_mask"].sum().item()
                    total_tokens = max(int(total_tokens), 1)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    _, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False
                    )
                    log_prob = log_prob.to(torch.float32)
                    mask = model_inputs["response_attention_mask"].to(log_prob.dtype)

                    # optional EOS filtering
                    if getattr(self.config, "ntp_ignore_eos", False) and hasattr(self, "tokenizer"):
                        eos = getattr(self.tokenizer, "eos_token_id", None)
                        if eos is not None:
                            not_eos = (model_inputs["responses"] != eos).to(mask.dtype)
                            mask = mask * not_eos

                    num = -(log_prob * mask).sum()
                    if getattr(self.config, "ntp_exact_token_mean", False):
                        loss = self.config.ntp_coeff * (num / total_tokens)
                    else:
                        den = mask.sum().clamp_min(1)
                        ntp_loss_tokmean = num / den
                        loss = self.config.ntp_coeff * (ntp_loss_tokmean / self.gradient_accumulation)

                    loss.backward()

                    append_to_dict(metrics, {
                        "critic/ntp_loss/token_mean": (num / mask.sum().clamp_min(1)).detach().item(),
                        "critic/ntp_tokens": mask.sum().detach().item(),
                    })

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
