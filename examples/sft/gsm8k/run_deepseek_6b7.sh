set -x

nproc_per_node=8

export MASTER_PORT=29500

# Shift the arguments so $@ refers to the rest
shift 2
#--standalone #--node_rank $NODE_RANK --rdzv_id "my_experiment" --rdzv_backend c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
torchrun --nnodes=2 --nproc_per_node=$nproc_per_node --node_rank $NODE_RANK --rdzv_id "my_experiment" --rdzv_backend c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/root/data/real_code \
    data.val_files=/root/data/real_code/test.parquet \
    data.response_key=text \
    data.max_length=1024 \
    data.train_batch_size=512 \
    data.truncation=right \
    optim.lr=2e-5 \
    optim.lr_scheduler=wsd \
    optim.weight_decay=0.1 \
    optim.warmup_steps_ratio=0 \
    +data.response_dict_keys=['text'] \
    data.micro_batch_size_per_gpu=32 \
    model.partial_pretrain=/root/.cache/huggingface/hub/models--01-ai--Yi-Coder-1.5B/snapshots/00e59e64f47d3c78e4cfbdd345888479797e8109 \
    model.use_liger=False \
    trainer.project_name=em-new \
    trainer.experiment_name=llama3b-cpt-sync-noassert \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@ \
    trainer.save_freq=2000 \
    trainer.test_freq=-1 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true