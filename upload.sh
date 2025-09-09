#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="/mnt/task_wrapper/user_output/artifacts/checkpoints/em-aug33/50-400-llama8b-10warmup-nopenalty-log-005lenpenalty-2sync-rerun"
DST_BASE="s3://afm-common-permanent/shenao_zhang/50-400-llama8b-10warmup-nopenalty-log-005lenpenalty-2sync-rerun"

for step in $(seq 1500 100 2100); do
  src="${SRC_BASE}/global_step_${step}/actor/huggingface/"
  dst="${DST_BASE}/global_step_${step}"
  echo "Uploading ${src} -> ${dst}"
  if [[ -d "$src" ]]; then
    aws s3 cp "$src" "$dst" --recursive
  else
    echo "WARNING: missing ${src}, skipping." >&2
  fi
done
