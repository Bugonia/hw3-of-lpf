#!/usr/bin/env bash
set -euo pipefail

PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
PROJECT_ROOT="${PROJECT_ROOT:-${PUBLIC_ROOT}/hw3-of-lpf}"
BASE_MODEL="${PUBLIC_ROOT}/models/Qwen3-VL-8B-Instruct"

if [[ -z "${MODEL_DIR:-}" ]]; then
  if [[ -d "${PROJECT_ROOT}/outputs/qwen3_vl_stage5_reasoning_merged" ]]; then
    export MODEL_DIR="${PROJECT_ROOT}/outputs/qwen3_vl_stage5_reasoning_merged"
  elif [[ -d "${PROJECT_ROOT}/outputs/qwen3_vl_stage4_repair_merged" ]]; then
    export MODEL_DIR="${PROJECT_ROOT}/outputs/qwen3_vl_stage4_repair_merged"
  elif [[ -d "${PROJECT_ROOT}/outputs/qwen3_vl_stage3_merged" ]]; then
    export MODEL_DIR="${PROJECT_ROOT}/outputs/qwen3_vl_stage3_merged"
  else
    export MODEL_DIR="$BASE_MODEL"
  fi
fi

export ADAPTER_DIR="${ADAPTER_DIR:-${PROJECT_ROOT}/outputs/qwen3_vl_stage6_vision_lora}"
export MERGED_DIR="${MERGED_DIR:-${PROJECT_ROOT}/outputs/qwen3_vl_stage6_vision_merged}"

cd "$PROJECT_ROOT"
echo "Merging adapter into base model: $MODEL_DIR"
bash scripts/merge_stage1_lora.sh
