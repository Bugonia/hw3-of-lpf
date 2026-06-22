#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
MODEL_DIR="${MODEL_DIR:-${PUBLIC_ROOT}/models/Qwen3-VL-8B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/rl_dpo_safe/qwen3_vl_dpo_safe_lora}"
MERGED_DIR="${MERGED_DIR:-outputs/rl_dpo_safe/qwen3_vl_dpo_safe_merged}"
LOG_DIR="${LOG_DIR:-outputs/rl_dpo_eval_logs}"
MIN_MERGE_FREE_GB="${MIN_MERGE_FREE_GB:-30}"
SKIP_MERGE="${SKIP_MERGE:-0}"

cd "$PROJECT_ROOT"

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] Adapter: $ADAPTER_DIR"
echo "[INFO] Merged model: $MERGED_DIR"

if [[ ! -d "$ADAPTER_DIR" ]]; then
  echo "[ERROR] Adapter directory not found: $ADAPTER_DIR" >&2
  exit 1
fi

merged_parent="$(dirname "$MERGED_DIR")"
mkdir -p "$merged_parent"
free_kb="$(df -Pk "$merged_parent" | awk 'NR==2 {print $4}')"
free_gb="$((free_kb / 1024 / 1024))"
echo "[INFO] Free space under $merged_parent: ${free_gb} GiB"

if [[ "$SKIP_MERGE" != "1" && "$free_gb" -lt "$MIN_MERGE_FREE_GB" ]]; then
  cat >&2 <<EOF
[ERROR] Not enough free disk space to merge Qwen3-VL.
Need at least ${MIN_MERGE_FREE_GB} GiB, found ${free_gb} GiB under:
  $merged_parent

The merged model is about 17 GiB plus write overhead. Use a filesystem with
more space, for example:

  MERGED_DIR=/tmp/qwen3_vl_dpo_safe_merged \\
  ADAPTER_DIR=$ADAPTER_DIR \\
  bash scripts/run_rl_dpo_eval.sh

If the merged model already exists and is complete, rerun with:

  SKIP_MERGE=1 MERGED_DIR=$MERGED_DIR bash scripts/run_rl_dpo_eval.sh
EOF
  exit 1
fi

if [[ "$SKIP_MERGE" == "1" ]]; then
  echo "[INFO] SKIP_MERGE=1, using existing merged model."
else
  python3 scripts/merge_qwen3_vl_lora.py \
    --base-model "$MODEL_DIR" \
    --adapter "$ADAPTER_DIR" \
    --output-dir "$MERGED_DIR"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 eval.py "$MERGED_DIR" \
  --split "${SPLIT:-dev}" \
  --tp "${TP:-1}" \
  --enforce-eager

echo "RL DPO eval finished. Merged model: $MERGED_DIR"
