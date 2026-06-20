#!/usr/bin/env bash
set -euo pipefail

PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
PROJECT_ROOT="${PROJECT_ROOT:-${PUBLIC_ROOT}/hw3-of-lpf}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/qwen3_vl_stage5_reasoning_lora}"
if [[ -z "${ADAPTER_NAME_OR_PATH:-}" ]]; then
  if [[ -d "${PROJECT_ROOT}/outputs/qwen3_vl_stage4_repair_lora" ]]; then
    export ADAPTER_NAME_OR_PATH="${PROJECT_ROOT}/outputs/qwen3_vl_stage4_repair_lora"
  else
    export ADAPTER_NAME_OR_PATH="${PROJECT_ROOT}/outputs/qwen3_vl_stage3_lora"
  fi
fi
export TRAIN_JSONL="${TRAIN_JSONL:-data/stage5_reasoning/sft_train.jsonl}"
export EVAL_JSONL="${EVAL_JSONL:-data/stage5_reasoning/sft_val.jsonl}"
export DATA_ROOT="${DATA_ROOT:-data/stage5_reasoning}"

# Conservative continuation from the best available adapter. The new labels are
# longer because they include point-check reasoning, so keep checkpoint saves off
# unless explicitly requested.
export LEARNING_RATE="${LEARNING_RATE:-3e-6}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-200}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
export EVAL_STEPS="${EVAL_STEPS:-200}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"

cd "$PROJECT_ROOT"
bash scripts/run_stage1_lora.sh
