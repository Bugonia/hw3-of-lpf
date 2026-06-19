#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/generative-large-model/public/hw3-of-lpf}"
PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
USER_GLOBAL="${USER_GLOBAL:-/inspire/hdd/global_user/zhongxiaoqiu-253108120179}"
MODEL_DIR="${MODEL_DIR:-${PUBLIC_ROOT}/models/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${PUBLIC_ROOT}/outputs/qwen3_vl_stage1_lora}"

export HF_HOME="${HF_HOME:-${USER_GLOBAL}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_GLOBAL}/.cache/pip}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$PIP_CACHE_DIR"
cd "$PROJECT_ROOT"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  python3 -m pip install -U \
    "git+https://github.com/huggingface/transformers" \
    accelerate peft bitsandbytes pillow tensorboard
fi
if [[ "${INSTALL_ONLY:-0}" == "1" ]]; then
  exit 0
fi

TRAIN_ARGS=(
  --model-name-or-path "$MODEL_DIR"
  --train-jsonl data/stage1_synth/sft_train.jsonl
  --eval-jsonl data/stage1_synth/sft_val.jsonl
  --data-root data/stage1_synth
  --output-dir "$OUTPUT_DIR"
  --max-length "${MAX_LENGTH:-4096}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}"
  --max-steps "${MAX_STEPS:--1}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --learning-rate "${LEARNING_RATE:-2e-4}"
  --warmup-ratio "${WARMUP_RATIO:-0.03}"
  --logging-steps "${LOGGING_STEPS:-10}"
  --eval-steps "${EVAL_STEPS:-100}"
  --save-steps "${SAVE_STEPS:-100}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}"
  --lora-r "${LORA_R:-16}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --lora-dropout "${LORA_DROPOUT:-0.05}"
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
  --optim "${OPTIM:-paged_adamw_8bit}"
)

if [[ "${LOAD_IN_4BIT:-1}" == "1" ]]; then
  TRAIN_ARGS+=(--load-in-4bit)
else
  TRAIN_ARGS+=(--no-load-in-4bit)
fi

if [[ "${DRY_RUN_BATCH:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--dry-run-batch --max-train-samples 4 --max-eval-samples 2)
fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  TRAIN_ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "${MAX_EVAL_SAMPLES:-}" ]]; then
  TRAIN_ARGS+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
fi

if [[ "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC_PER_NODE}" scripts/train_qwen3_vl_lora.py "${TRAIN_ARGS[@]}"
else
  python3 scripts/train_qwen3_vl_lora.py "${TRAIN_ARGS[@]}"
fi
