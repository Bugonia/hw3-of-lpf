#!/usr/bin/env bash
set -euo pipefail

# Guarded long DPO/RL training with all heavy artifacts stored under the user's
# global storage quota instead of the nearly-full project filesystem.

GLOBAL_ROOT="${GLOBAL_ROOT:-/inspire/hdd/global_user/yuwenye-253108120175}"
RUN_NAME="${RUN_NAME:-hw3_rl_guarded_$(date +%Y%m%d_%H%M%S)}"
GLOBAL_RUN_DIR="${GLOBAL_RUN_DIR:-${GLOBAL_ROOT}/hw3_rl_runs/${RUN_NAME}}"

mkdir -p "$GLOBAL_RUN_DIR"

export RUN_DIR="${RUN_DIR:-${GLOBAL_RUN_DIR}}"
export TMP_MERGE_ROOT="${TMP_MERGE_ROOT:-${GLOBAL_RUN_DIR}/merged_models}"
export LOG_DIR="${LOG_DIR:-${GLOBAL_RUN_DIR}/logs}"

# Recommended defaults for the 140G single-GPU server.
export PHASES="${PHASES:-8}"
export PHASE_STEPS="${PHASE_STEPS:-10}"
export MAX_PREF_SAMPLES="${MAX_PREF_SAMPLES:-3000}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export LEARNING_RATE="${LEARNING_RATE:-2e-7}"
export DPO_BETA="${DPO_BETA:-0.02}"
export SFT_LOSS_COEF="${SFT_LOSS_COEF:-0.05}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
export REJECTION_MODE="${REJECTION_MODE:-hardest}"

echo "[INFO] Global guarded RL run directory: $GLOBAL_RUN_DIR"
echo "[INFO] Heavy outputs will be stored outside the project filesystem."

bash scripts/run_rl_dpo_guarded_long_train.sh

echo "[INFO] Finished global guarded RL run."
echo "[INFO] Run directory: $GLOBAL_RUN_DIR"
echo "[INFO] Status log: ${GLOBAL_RUN_DIR}/guarded_status.jsonl"
