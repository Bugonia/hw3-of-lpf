#!/usr/bin/env bash
set -euo pipefail

# Guarded long DPO/RL training with all heavy artifacts stored under the user's
# global storage quota instead of the nearly-full project filesystem.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GLOBAL_ROOT="${GLOBAL_ROOT:-/inspire/hdd/global_user/yuwenye-253108120175}"
RUN_NAME="${RUN_NAME:-hw3_rl_guarded_$(date +%Y%m%d_%H%M%S)}"
GLOBAL_RUN_DIR="${GLOBAL_RUN_DIR:-${GLOBAL_ROOT}/hw3_rl_runs/${RUN_NAME}}"

mkdir -p "$GLOBAL_RUN_DIR"

export RUN_LOG_FILE="${RUN_LOG_FILE:-${GLOBAL_RUN_DIR}/run_full.log}"
if [[ "${LOG_TO_STDOUT:-1}" == "1" ]]; then
  exec > >(tee -a "$RUN_LOG_FILE") 2>&1
else
  exec >> "$RUN_LOG_FILE" 2>&1
fi

export RUN_DIR="${RUN_DIR:-${GLOBAL_RUN_DIR}}"
export TMP_MERGE_ROOT="${TMP_MERGE_ROOT:-${GLOBAL_RUN_DIR}/merged_models}"
export LOG_DIR="${LOG_DIR:-${GLOBAL_RUN_DIR}/logs}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${GLOBAL_RUN_DIR}/eval_outputs}"

# Keep library caches and temporary files off the nearly-full project filesystem.
CACHE_ROOT="${CACHE_ROOT:-${GLOBAL_ROOT}/.cache}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export XDG_CACHE_HOME="${CACHE_ROOT}"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
# vLLM/ZMQ creates IPC sockets under TMPDIR. Unix socket paths are limited
# to about 107 bytes, so keep this path short even though checkpoints live
# under GLOBAL_RUN_DIR.
SAFE_RUN_NAME="$(printf '%s' "$RUN_NAME" | tr -c 'A-Za-z0-9_.-' '_')"
export TMPDIR="${TMPDIR:-/tmp/hw3rl_${SAFE_RUN_NAME}_$$}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
mkdir -p \
  "$LOG_DIR" \
  "$EVAL_OUTPUT_ROOT" \
  "$TMP_MERGE_ROOT" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_MODULES_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$VLLM_CACHE_ROOT" \
  "$TMPDIR"

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

on_exit() {
  code=$?
  echo "[INFO] Exit code: $code"
  echo "[INFO] Finished at: $(date -Is)"
  echo "[INFO] Full log: $RUN_LOG_FILE"
  echo "[INFO] Disk usage for run dir:"
  du -sh "$GLOBAL_RUN_DIR" 2>/dev/null || true
  echo "[INFO] Filesystem status:"
  df -h "$GLOBAL_ROOT" "$PROJECT_ROOT" 2>/dev/null || df -h "$GLOBAL_ROOT" 2>/dev/null || true
  if [[ "$TMPDIR" == /tmp/hw3rl_* ]]; then
    rm -rf "$TMPDIR" 2>/dev/null || true
  fi
}
trap on_exit EXIT

echo "[INFO] Global guarded RL run directory: $GLOBAL_RUN_DIR"
echo "[INFO] Heavy outputs will be stored outside the project filesystem."
echo "[INFO] Full log: $RUN_LOG_FILE"
echo "[INFO] Started at: $(date -Is)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Key output paths:"
echo "  RUN_DIR=$RUN_DIR"
echo "  TMP_MERGE_ROOT=$TMP_MERGE_ROOT"
echo "  EVAL_OUTPUT_ROOT=$EVAL_OUTPUT_ROOT"
echo "  LOG_DIR=$LOG_DIR"
echo "  TMPDIR=$TMPDIR"
echo "[INFO] Cache paths:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "  TORCH_HOME=$TORCH_HOME"
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "  TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
echo "  VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"
echo "[INFO] Filesystem status before run:"
df -h "$GLOBAL_ROOT" "$PROJECT_ROOT" 2>/dev/null || df -h "$GLOBAL_ROOT" 2>/dev/null || true
echo "[INFO] CUDA status before run:"
nvidia-smi 2>/dev/null || true
echo "[INFO] Training config:"
echo "  PHASES=$PHASES"
echo "  PHASE_STEPS=$PHASE_STEPS"
echo "  MAX_PREF_SAMPLES=$MAX_PREF_SAMPLES"
echo "  PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
echo "  LEARNING_RATE=$LEARNING_RATE"
echo "  DPO_BETA=$DPO_BETA"
echo "  SFT_LOSS_COEF=$SFT_LOSS_COEF"
echo "  WARMUP_RATIO=$WARMUP_RATIO"
echo "  REJECTION_MODE=$REJECTION_MODE"

bash scripts/run_rl_dpo_guarded_long_train.sh

echo "[INFO] Finished global guarded RL run."
echo "[INFO] Run directory: $GLOBAL_RUN_DIR"
echo "[INFO] Status log: ${GLOBAL_RUN_DIR}/guarded_status.jsonl"
