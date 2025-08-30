# run on 4xH100
# make sure your current working directory is the root of the project

set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

ray job submit \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo \
        --config-path="$CONFIG_PATH" \
        --config-name='gsm8k_multiturn_grpo' \
        algorithm.adv_estimator=grpo \
        data.train_batch_size=1024 \
        data.max_prompt_length=128 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        data.filter_overlong_prompts_workers=48 \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B \
        +actor_rollout_ref.actor.ntp_coeff=1.0 \
        actor_rollout_ref.actor.optim.lr=2e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        +actor_rollout_ref.actor.ntp_mini_batch_size=512 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=24 \
        +actor_rollout_ref.actor.ntp_micro_batch_size_per_gpu=64 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=80 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=80 \
        +actor_rollout_ref.ref.logr=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=3 \
        actor_rollout_ref.rollout.temperature=1.0 \
        +actor_rollout_ref.rollout.per_turn_response_length=16 \
        +actor_rollout_ref.rollout.max_code_lines=32 \
        actor_rollout_ref.rollout.response_length=1024 \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='em-aug30-debug' \
        trainer.experiment_name='40-400-qwen-40warmup-nopenalty-log-nolenpenalty' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.val_before_train=False \
        trainer.save_freq=200 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        +trainer.q_steps=40 \
        +trainer.ref_update_freq=400 \
        data.train_files=/mnt/task_runtime/opencoder_post.parquet \
        data.val_files=s3://afm-common-permanent/shenao_zhang/sync_code_aug29/test.parquet \
        actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/gsm8k_interaction_config.yaml" \
        actor_rollout_ref.rollout.multi_turn.max_user_turns=1 \
        $@
        #actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \