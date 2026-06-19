#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/generative-large-model/public/hw3-of-lpf}"
PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
USER_GLOBAL="${USER_GLOBAL:-/inspire/hdd/global_user/zhongxiaoqiu-253108120179}"
MODEL_DIR="${MODEL_DIR:-${PUBLIC_ROOT}/models/Qwen3-VL-8B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-${PUBLIC_ROOT}/outputs/qwen3_vl_stage1_lora}"
MERGED_DIR="${MERGED_DIR:-${PUBLIC_ROOT}/outputs/qwen3_vl_stage1_merged}"

export HF_HOME="${HF_HOME:-${USER_GLOBAL}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_GLOBAL}/.cache/pip}"

cd "$PROJECT_ROOT"
python3 scripts/merge_qwen3_vl_lora.py \
  --base-model "$MODEL_DIR" \
  --adapter "$ADAPTER_DIR" \
  --output-dir "$MERGED_DIR"

cp prompt_example.txt "$MERGED_DIR/prompt.txt"
chmod -R a+rX "$MERGED_DIR"
du -sh "$MERGED_DIR"
