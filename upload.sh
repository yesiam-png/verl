#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="/mnt/task_wrapper/user_output/artifacts/checkpoints/em-new-debug/openandsyn-codegemma-ntponly-1sync-rerun2"
DST_BASE="s3://afm-common-permanent/shenao_zhang/openandsyn-codegemma-ntponly-1sync-rerun2"

for step in $(seq 1999 200 2400); do
  src="${SRC_BASE}/global_step_${step}/actor/huggingface/"
  dst="${DST_BASE}/global_step_${step}"
  echo "Uploading ${src} -> ${dst}"
  if [[ -d "$src" ]]; then
    aws s3 cp "$src" "$dst" --recursive
  else
    echo "WARNING: missing ${src}, skipping." >&2
  fi
done
