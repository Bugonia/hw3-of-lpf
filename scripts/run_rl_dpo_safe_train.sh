#!/usr/bin/env bash
set -euo pipefail

# Conservative DPO/RL recipe after the 100-step aggressive run regressed dev
# from 68.0% to 38.7%. This wrapper intentionally uses harder negatives,
# lower learning rate, lower beta, fewer updates, and a larger effective batch.

export OUT_DIR="${OUT_DIR:-outputs/rl_dpo_safe}"
export ADAPTER_OUT="${ADAPTER_OUT:-${OUT_DIR}/qwen3_vl_dpo_safe_lora}"
export PREF_JSONL="${PREF_JSONL:-${OUT_DIR}/data/preferences.jsonl}"

# Keep the preference set moderate and verifier-hard. The first failed run used
# very easy generic negatives, which made the DPO objective too blunt.
export MAX_PREF_SAMPLES="${MAX_PREF_SAMPLES:-2000}"
export REJECTION_MODE="${REJECTION_MODE:-hardest}"

# Safer optimization defaults for a single 140G GPU. Effective batch remains
# 4 * 8 * 1 = 32, but larger micro-batches reduce accumulation overhead.
export MAX_STEPS="${MAX_STEPS:-30}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-5e-7}"
export DPO_BETA="${DPO_BETA:-0.03}"
export SFT_LOSS_COEF="${SFT_LOSS_COEF:-0.03}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
export LOGGING_STEPS="${LOGGING_STEPS:-5}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"

bash scripts/run_rl_dpo_train.sh

echo "Safe RL DPO training finished."
echo "Adapter: ${ADAPTER_OUT}"
echo "Suggested eval:"
echo "  ADAPTER_DIR=${ADAPTER_OUT} MERGED_DIR=${OUT_DIR}/qwen3_vl_dpo_safe_merged bash scripts/run_rl_dpo_eval.sh"
