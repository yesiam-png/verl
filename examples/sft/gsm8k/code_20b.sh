set -x

nproc_per_node=8

export MASTER_ADDR=240.62.237.180
export MASTER_PORT=29500

# Shift the arguments so $@ refers to the rest
shift 2
#--standalone #--node_rank $NODE_RANK --rdzv_id "my_experiment" --rdzv_backend c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
torchrun --nnodes=8 --nproc_per_node=$nproc_per_node --node_rank $NODE_RANK --rdzv_id "my_experiment" --rdzv_backend c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=s3://afm-common-permanent/shenao_zhang/small_20b_python_data/train \
    data.val_files=s3://afm-common-permanent/shenao_zhang/small_20b_python_data/test \
    data.response_key=text \
    data.max_length=8192 \
    data.train_batch_size=512 \
    data.truncation=right \
    optim.lr=5e-5 \
    optim.lr_scheduler=wsd \
    optim.weight_decay=0.1 \
    optim.warmup_steps_ratio=0 \
    +data.response_dict_keys=['text'] \
    data.micro_batch_size_per_gpu=16 \
    model.partial_pretrain=ZhangShenao/Llama-3.2-3B \
    model.use_liger=True \
    trainer.project_name=llama-cpt \
    trainer.experiment_name=llama3b-cpt-real20b \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@ \
    trainer.save_freq=2000 \
    trainer.test_freq=-1 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true