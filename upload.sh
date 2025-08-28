#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="/mnt/task_wrapper/user_output/artifacts/checkpoints/em-aug24/40-400-qwen-10warmup-005penalty-log-nolenpenalty"
DST_BASE="s3://afm-common-permanent/shenao_zhang/temp_ckpts/Aug28/40-400-qwen-10warmup-005penalty-log-nolenpenalty"

for step in $(seq 200 200 2400); do
  src="${SRC_BASE}/global_step_${step}/actor/huggingface/"
  dst="${DST_BASE}/global_step_${step}"
  echo "Uploading ${src} -> ${dst}"
  if [[ -d "$src" ]]; then
    aws s3 cp "$src" "$dst" --recursive
  else
    echo "WARNING: missing ${src}, skipping." >&2
  fi
done
