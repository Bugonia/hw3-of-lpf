#!/usr/bin/env bash
set -euo pipefail

PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
PROJECT_ROOT="${PROJECT_ROOT:-${PUBLIC_ROOT}/hw3-of-lpf}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/qwen3_vl_stage2_lora}"
export ADAPTER_NAME_OR_PATH="${ADAPTER_NAME_OR_PATH:-${PROJECT_ROOT}/outputs/qwen3_vl_stage1_lora}"
export TRAIN_JSONL="${TRAIN_JSONL:-data/stage2_targeted/sft_train.jsonl}"
export EVAL_JSONL="${EVAL_JSONL:-data/stage2_targeted/sft_val.jsonl}"
export DATA_ROOT="${DATA_ROOT:-data/stage2_targeted}"

# Continuing an already useful adapter benefits from a smaller default LR.
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-200}"
export EVAL_STEPS="${EVAL_STEPS:-200}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

cd "$PROJECT_ROOT"
bash scripts/run_stage1_lora.sh
