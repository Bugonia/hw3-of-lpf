#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
MODEL_DIR="${MODEL_DIR:-${PUBLIC_ROOT}/models/Qwen3-VL-8B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/rl_dpo/qwen3_vl_dpo_lora}"
MERGED_DIR="${MERGED_DIR:-outputs/rl_dpo/qwen3_vl_dpo_merged}"

cd "$PROJECT_ROOT"

python3 scripts/merge_qwen3_vl_lora.py \
  --base-model "$MODEL_DIR" \
  --adapter "$ADAPTER_DIR" \
  --output-dir "$MERGED_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 eval.py "$MERGED_DIR" \
  --split "${SPLIT:-dev}" \
  --tp "${TP:-1}" \
  --enforce-eager

echo "RL DPO eval finished. Merged model: $MERGED_DIR"
