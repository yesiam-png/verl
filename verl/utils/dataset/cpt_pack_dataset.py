# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from tqdm import tqdm

class PackedCPTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
      #  prompt_key = config.get("prompt_key", "prompt")
      #  prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm
        if parquet_files.endswith("train"):
            parquet_files = [parquet_files[:-5] + f"000_{i:05d}.parquet" for i in range(100)]
           # self.split = "train"
        elif parquet_files.endswith("test"):
            parquet_files = [parquet_files[:-4] + f"000_{i:05d}.parquet" for i in range(1)]
           # self.split = "test"
        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

  #      self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
   #     self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

       # self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
#        self.prompts = self.dataframe[self.prompt_key]
      #  for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
      #      try:
      #          self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
      #      except Exception:
      #          print(f"self.prompts={self.prompts}")
      #          raise
      #  if isinstance(self.prompts, pd.DataFrame):
      #      self.prompts = self.prompts.squeeze()
     #   self.prompts = self.prompts.tolist()
       # responses = self.dataframe#[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.dataframe.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.dataframe}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
       # print(len(list(self.responses)), self.responses[0], "sssdss")
        #if self.split == "test":
        self.responses = self.responses.tolist()

        # Newly added
        self.packed_examples = self.pack_sequences(self.responses)
        print(f"Created {len(self.packed_examples)} packed examples from {len(self.responses)} original sequences")

    def _finalize_packed_sequence(
        self,
        packed_data,
        input_ids: List[int],
        attn_mask: List[int],
        pos_ids: List[int],
        seq_boundaries,
    ) -> None:
        L = len(input_ids)
        # 1D → 4D padding mask: [1,1,1,L]
        pad_mask = torch.tensor(attn_mask, dtype=torch.float32)[None, None, None, :]

        # Build block-diagonal causal mask via torch.block_diag
        lengths = [b["end"] - b["start"] for b in seq_boundaries]
        blocks = [torch.tril(torch.ones((l, l), dtype=torch.float32)) for l in lengths]
        block = torch.block_diag(*blocks)  # [L, L]
        attention_mask = pad_mask * block[None, None, :, :]  # [1,1,L,L]

        # Loss mask: 1 on real tokens, 0 on pads; zero out EOS tokens
        loss_mask = torch.tensor(attn_mask, dtype=torch.long)
        eos_indices = [b["end"] - 1 for b in seq_boundaries]
        loss_mask[eos_indices] = 0

        example = {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": attention_mask,               # 4D float mask
            "position_ids":   torch.tensor(pos_ids,        dtype=torch.long),
            "loss_mask":      loss_mask,                   # 1D long mask
        }
        packed_data.append(example)

    def pack_sequences(self, sequences: List[str]):
        """
        Tokenizes and packs a list of raw text sequences into
        fixed-length examples ready for model training.
        """
        packed_data = []
        cur_ids: List[int] = []
        cur_attn: List[int] = []
        cur_pos: List[int] = []
        cur_bounds = []
        batch_size = 512

        for i in tqdm(range(0, len(sequences), batch_size)):
            batch = sequences[i : i + batch_size]
            # Fast batch tokenization
            batch_texts = [text + self.tokenizer.eos_token for text in batch]
            enc = self.tokenizer(batch_texts, add_special_tokens=False)

            for j, seq_input_ids in enumerate(enc["input_ids"]):
                seq_idx = i + j
                # Apply truncation if necessary
                if len(seq_input_ids) > self.max_length:
                    if self.truncation == "left":
                        seq_input_ids = seq_input_ids[-self.max_length:]
                    elif self.truncation == "right":
                        seq_input_ids = seq_input_ids[: self.max_length]
                    else:
                        raise ValueError(
                            f"Sequence length {len(seq_input_ids)} exceeds max_length={self.max_length}"
                        )

                length = len(seq_input_ids)
                # If overflow, finalize current pack and start a new one
                if len(cur_ids) + length > self.max_length:
                    if cur_ids:
                        self._finalize_packed_sequence(
                            packed_data,
                            cur_ids,
                            cur_attn,
                            cur_pos,
                            cur_bounds,
                        )
                    cur_ids, cur_attn, cur_pos, cur_bounds = [], [], [], []

                # Record boundaries and extend buffers
                start = len(cur_ids)
                end = start + length
                cur_bounds.append({"start": start, "end": end, "seq_id": seq_idx})
                cur_ids.extend(seq_input_ids)
                cur_attn.extend([1] * length)
                cur_pos.extend(list(range(length)))

        # Finalize the last pack
        if cur_ids:
            self._finalize_packed_sequence(
                packed_data,
                cur_ids,
                cur_attn,
                cur_pos,
                cur_bounds,
            )

        return packed_data

    def __len__(self):
        return len(self.packed_examples)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

     #   prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
     #   prompt_chat = [{"role": "user", "content": prompt}]

        # string
     #   prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        # tokenize
      #  prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
      #  prompt_ids = prompt_ids_output["input_ids"][0]
      #  prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        attention_mask = response_ids_output["attention_mask"][0]

      #  prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

       # input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
       # attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = response_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=response_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            response_ids = torch.cat((response_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                response_ids = response_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                response_ids = response_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
       # if prompt_length > 1:
            # mask out prompt for SFT.
        #    loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": response_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
