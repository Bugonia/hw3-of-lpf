#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
MODEL_DIR="${MODEL_DIR:-${PUBLIC_ROOT}/models/Qwen3-VL-8B-Instruct}"
BEST_ADAPTER="${BEST_ADAPTER:-${PUBLIC_ROOT}/hw3-of-lpf-best/qwen3_vl_v4_targeted_20260621/adapter}"
SAMPLES="${SAMPLES:-${PUBLIC_ROOT}/hw3-of-lpf/data/sft_v2/samples_train.jsonl}"
DATA_ROOT="${DATA_ROOT:-$(dirname "$SAMPLES")}"
BASELINE_RESULTS="${BASELINE_RESULTS:-}"
OUT_DIR="${OUT_DIR:-outputs/rl_dpo}"
PREF_JSONL="${PREF_JSONL:-${OUT_DIR}/data/preferences.jsonl}"

cd "$PROJECT_ROOT"

mkdir -p "$OUT_DIR/data"

GEN_ARGS=(
  --samples "$SAMPLES"
  --data-root "$DATA_ROOT"
  --out "$PREF_JSONL"
  --seed "${SEED:-20260622}"
)
if [[ -n "${MAX_PREF_SAMPLES:-}" ]]; then
  GEN_ARGS+=(--max-samples "$MAX_PREF_SAMPLES")
fi
if [[ -n "$BASELINE_RESULTS" && -f "$BASELINE_RESULTS" ]]; then
  GEN_ARGS+=(--baseline-results "$BASELINE_RESULTS")
fi
GEN_ARGS+=(--rejection-mode "${REJECTION_MODE:-hardest}")

python3 scripts/generate_dpo_preference_data.py "${GEN_ARGS[@]}"

python3 scripts/train_qwen3_vl_dpo_lora.py \
  --model-name-or-path "$MODEL_DIR" \
  --adapter-name-or-path "$BEST_ADAPTER" \
  --train-jsonl "$PREF_JSONL" \
  --output-dir "${ADAPTER_OUT:-${OUT_DIR}/qwen3_vl_dpo_lora}" \
  --max-length "${MAX_LENGTH:-4096}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --max-steps "${MAX_STEPS:--1}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning-rate "${LEARNING_RATE:-5e-6}" \
  --warmup-ratio "${WARMUP_RATIO:-0.03}" \
  --beta "${DPO_BETA:-0.1}" \
  --sft-loss-coef "${SFT_LOSS_COEF:-0.0}" \
  --logging-steps "${LOGGING_STEPS:-10}" \
  --save-steps "${SAVE_STEPS:-200}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"

echo "RL DPO training finished. Adapter: ${ADAPTER_OUT:-${OUT_DIR}/qwen3_vl_dpo_lora}"
