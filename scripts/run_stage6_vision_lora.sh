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

# Start a fresh higher-rank adapter on top of the merged best model. Continuing
# an existing LoRA adapter would keep its old rank and target modules.
unset ADAPTER_NAME_OR_PATH
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/qwen3_vl_stage6_vision_lora}"
export TRAIN_JSONL="${TRAIN_JSONL:-data/stage5_reasoning/sft_train.jsonl}"
export EVAL_JSONL="${EVAL_JSONL:-data/stage5_reasoning/sft_val.jsonl}"
export DATA_ROOT="${DATA_ROOT:-data/stage5_reasoning}"

export LORA_R="${LORA_R:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export FREEZE_VISION_LORA="${FREEZE_VISION_LORA:-0}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,qkv}"

export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-200}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
export EVAL_STEPS="${EVAL_STEPS:-200}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"

cd "$PROJECT_ROOT"
echo "Base model for fresh vision-capable LoRA: $MODEL_DIR"
bash scripts/run_stage1_lora.sh
