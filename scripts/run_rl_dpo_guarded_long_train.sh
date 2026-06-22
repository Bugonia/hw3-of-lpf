#!/usr/bin/env bash
set -euo pipefail

# Long-running but guarded RL/DPO training. It trains in small phases and runs
# dev eval after every phase. A phase is accepted only if it improves the best
# dev acc@0.99 seen so far; otherwise the next phase restarts from the best
# accepted adapter. If the score drops below MIN_KEEP_ACC, the script stops.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
START_ADAPTER="${START_ADAPTER:-${PUBLIC_ROOT}/hw3-of-lpf-best/qwen3_vl_v4_targeted_20260621/adapter}"
RUN_DIR="${RUN_DIR:-outputs/rl_dpo_guarded}"
TMP_MERGE_ROOT="${TMP_MERGE_ROOT:-/tmp}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${RUN_DIR}/eval_outputs}"

# Current best SFT checkpoint is 68.0 acc@0.99 on dev.
BASELINE_ACC="${BASELINE_ACC:-0.68}"
MIN_KEEP_ACC="${MIN_KEEP_ACC:-0.68}"
MIN_IMPROVEMENT="${MIN_IMPROVEMENT:-0.001}"

# Phase schedule. On a 140G single GPU, batch 8 x accumulation 4 keeps the
# effective batch at 32 while using memory better than micro-batch 1.
PHASES="${PHASES:-8}"
PHASE_STEPS="${PHASE_STEPS:-10}"
MAX_PREF_SAMPLES="${MAX_PREF_SAMPLES:-3000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-7}"
DPO_BETA="${DPO_BETA:-0.02}"
SFT_LOSS_COEF="${SFT_LOSS_COEF:-0.05}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
REJECTION_MODE="${REJECTION_MODE:-hardest}"

cd "$PROJECT_ROOT"
mkdir -p "$RUN_DIR"

best_adapter="$START_ADAPTER"
best_acc="$BASELINE_ACC"
status_json="${RUN_DIR}/guarded_status.jsonl"
: > "$status_json"

echo "[INFO] Starting guarded DPO run"
echo "[INFO] start_adapter=$START_ADAPTER"
echo "[INFO] run_dir=$RUN_DIR"
echo "[INFO] baseline_acc=$BASELINE_ACC min_keep_acc=$MIN_KEEP_ACC"
echo "[INFO] phase defaults: phases=$PHASES phase_steps=$PHASE_STEPS batch=${PER_DEVICE_TRAIN_BATCH_SIZE}x${GRADIENT_ACCUMULATION_STEPS} lr=$LEARNING_RATE beta=$DPO_BETA sft=$SFT_LOSS_COEF"

for phase in $(seq 1 "$PHASES"); do
  phase_dir="${RUN_DIR}/phase_${phase}"
  adapter_out="${phase_dir}/adapter"
  merged_dir="${TMP_MERGE_ROOT}/qwen3_vl_dpo_guarded_phase_${phase}_merged"
  mkdir -p "$phase_dir"

  echo "[INFO] ===== Phase ${phase}/${PHASES} ====="
  echo "[INFO] training_from=$best_adapter"
  echo "[INFO] adapter_out=$adapter_out"

  BEST_ADAPTER="$best_adapter" \
  OUT_DIR="$phase_dir" \
  ADAPTER_OUT="$adapter_out" \
  PREF_JSONL="${phase_dir}/data/preferences.jsonl" \
  MAX_PREF_SAMPLES="$MAX_PREF_SAMPLES" \
  REJECTION_MODE="$REJECTION_MODE" \
  MAX_STEPS="$PHASE_STEPS" \
  PER_DEVICE_TRAIN_BATCH_SIZE="$PER_DEVICE_TRAIN_BATCH_SIZE" \
  GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
  LEARNING_RATE="$LEARNING_RATE" \
  DPO_BETA="$DPO_BETA" \
  SFT_LOSS_COEF="$SFT_LOSS_COEF" \
  WARMUP_RATIO="$WARMUP_RATIO" \
  LOGGING_STEPS="${LOGGING_STEPS:-5}" \
  bash scripts/run_rl_dpo_train.sh

  rm -rf "$merged_dir"
  ADAPTER_DIR="$adapter_out" \
  MERGED_DIR="$merged_dir" \
  LOG_DIR="${phase_dir}/eval_logs" \
  EVAL_OUTPUT_ROOT="$EVAL_OUTPUT_ROOT" \
  bash scripts/run_rl_dpo_eval.sh

  summary_path="${EVAL_OUTPUT_ROOT}/$(basename "$merged_dir")/eval_summary_dev.json"
  if [[ ! -f "$summary_path" ]]; then
    echo "[ERROR] Eval summary not found: $summary_path" >&2
    exit 1
  fi

  decision_json="$(python3 - "$summary_path" "$phase" "$best_acc" "$MIN_KEEP_ACC" "$MIN_IMPROVEMENT" "$adapter_out" <<'PY'
import json
import sys

summary_path, phase, best_acc, min_keep, min_improve, adapter = sys.argv[1:]
summary = json.load(open(summary_path))
acc = float(summary["acc@0.99"])
best = float(best_acc)
min_keep = float(min_keep)
min_improve = float(min_improve)
if acc < min_keep:
    decision = "stop_drop"
elif acc >= best + min_improve:
    decision = "accept"
else:
    decision = "reject_no_improve"
print(json.dumps({
    "phase": int(phase),
    "acc@0.99": acc,
    "acc@0.95": summary.get("acc@0.95"),
    "acc@0.9": summary.get("acc@0.9"),
    "acc@0.8": summary.get("acc@0.8"),
    "mean_r2": summary.get("mean_r2"),
    "median_r2": summary.get("median_r2"),
    "best_before": best,
    "decision": decision,
    "adapter": adapter,
}))
PY
)"
  echo "$decision_json" | tee -a "$status_json"
  decision="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["decision"])' "$decision_json")"
  phase_acc="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["acc@0.99"])' "$decision_json")"

  if [[ "$decision" == "accept" ]]; then
    best_adapter="$adapter_out"
    best_acc="$phase_acc"
    echo "[INFO] Accepted phase $phase. New best acc@0.99=$best_acc"
  elif [[ "$decision" == "stop_drop" ]]; then
    echo "[WARN] Phase $phase dropped below MIN_KEEP_ACC=$MIN_KEEP_ACC. Stopping guarded run."
    echo "[INFO] Best adapter remains: $best_adapter"
    exit 0
  else
    echo "[INFO] Phase $phase did not improve. Keeping best adapter: $best_adapter"
  fi
done

echo "[INFO] Guarded run finished."
echo "[INFO] Best acc@0.99=$best_acc"
echo "[INFO] Best adapter=$best_adapter"
echo "[INFO] Status log=$status_json"
