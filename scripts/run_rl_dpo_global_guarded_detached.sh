#!/usr/bin/env bash
set -euo pipefail

# Start the global guarded RL run in the background. Use this when the terminal
# or SSH session is unstable; the full training log stays in the run directory.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GLOBAL_ROOT="${GLOBAL_ROOT:-/inspire/hdd/global_user/yuwenye-253108120175}"
RUN_NAME="${RUN_NAME:-hw3_rl_guarded_$(date +%Y%m%d_%H%M%S)}"
GLOBAL_RUN_DIR="${GLOBAL_RUN_DIR:-${GLOBAL_ROOT}/hw3_rl_runs/${RUN_NAME}}"

mkdir -p "$GLOBAL_RUN_DIR"

export PROJECT_ROOT
export GLOBAL_ROOT
export RUN_NAME
export GLOBAL_RUN_DIR
export RUN_LOG_FILE="${RUN_LOG_FILE:-${GLOBAL_RUN_DIR}/run_full.log}"
export LOG_TO_STDOUT=0

launcher_log="${GLOBAL_RUN_DIR}/launcher.log"

cd "$PROJECT_ROOT"

nohup bash scripts/run_rl_dpo_global_guarded_train.sh > "$launcher_log" 2>&1 &
pid=$!
echo "$pid" > "${GLOBAL_RUN_DIR}/pid.txt"

cat <<EOF
[INFO] Started detached guarded RL run.
[INFO] PID: $pid
[INFO] Run directory: $GLOBAL_RUN_DIR
[INFO] Launcher log: $launcher_log
[INFO] Full training log: $RUN_LOG_FILE

Useful commands:
  tail -f "$RUN_LOG_FILE"
  tail -f "${GLOBAL_RUN_DIR}/guarded_status.jsonl"
  ps -p $pid -o pid,stat,etime,cmd
EOF
