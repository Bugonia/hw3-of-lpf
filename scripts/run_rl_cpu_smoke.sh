#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-outputs/rl_cpu_smoke}"
SAMPLES="${SAMPLES:-data/task/dev/samples.jsonl}"
BASELINE_RESULTS="${BASELINE_RESULTS:-/inspire/hdd/project/generative-large-model/public/hw3-of-lpf-best/qwen3_vl_v4_targeted_20260621/eval_outputs/eval_results_dev.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
MAX_CANDIDATES="${MAX_CANDIDATES:-12}"
EPOCHS="${EPOCHS:-80}"
SEED="${SEED:-20260622}"
DROP_TOLERANCE="${DROP_TOLERANCE:-0.02}"

cd "$PROJECT_ROOT"

python3 - <<'PY'
import importlib
missing = []
for name in ("numpy", "torch"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, str(exc)))
if missing:
    for name, exc in missing:
        print(f"missing dependency: {name}: {exc}")
    raise SystemExit(1)
PY

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

GEN_ARGS=(
  --samples "$SAMPLES"
  --out-dir "$OUT_DIR/data"
  --max-samples "$MAX_SAMPLES"
  --max-candidates "$MAX_CANDIDATES"
  --seed "$SEED"
)

if [[ -f "$BASELINE_RESULTS" ]]; then
  GEN_ARGS+=(--baseline-results "$BASELINE_RESULTS")
else
  echo "Baseline results not found, running reward-improvement smoke only: $BASELINE_RESULTS"
fi

python3 scripts/generate_rl_candidate_data.py "${GEN_ARGS[@]}"

python3 scripts/train_rl_candidate_policy.py \
  --train-jsonl "$OUT_DIR/data/rl_train.jsonl" \
  --eval-jsonl "$OUT_DIR/data/rl_eval.jsonl" \
  --output-dir "$OUT_DIR/policy" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --device cpu

python3 scripts/evaluate_rl_candidate_policy.py \
  --checkpoint "$OUT_DIR/policy/candidate_policy.pt" \
  --eval-jsonl "$OUT_DIR/data/rl_eval.jsonl" \
  --output-dir "$OUT_DIR/eval" \
  --device cpu

export OUT_DIR DROP_TOLERANCE
python3 - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
drop_tolerance = float(os.environ["DROP_TOLERANCE"])
summary_path = out_dir / "policy" / "train_summary.json"
eval_path = out_dir / "eval" / "eval_summary.json"
manifest_path = out_dir / "data" / "manifest.json"
summary = json.loads(summary_path.read_text())
eval_summary = json.loads(eval_path.read_text())
manifest = json.loads(manifest_path.read_text())

before = summary["before_eval"]["mean_reward"]
after = summary["after_eval"]["mean_reward"]
acc = eval_summary["acc@0.99"]
baseline_eval = manifest.get("baseline_eval") or {}
baseline_acc = baseline_eval.get("acc@0.99")
baseline_reward = baseline_eval.get("mean_reward")

print(
    "CPU RL smoke gate: "
    f"before_eval_mean_reward={before:.4f}, "
    f"after_eval_mean_reward={after:.4f}, "
    f"eval_acc@0.99={acc:.4f}, "
    f"baseline_acc@0.99={baseline_acc if baseline_acc is not None else 'n/a'}, "
    f"baseline_mean_reward={baseline_reward if baseline_reward is not None else 'n/a'}"
)
if after <= before:
    raise SystemExit("RL smoke failed: eval mean reward did not improve")
if acc < 0.50:
    raise SystemExit("RL smoke failed: eval acc@0.99 below 0.50")
if baseline_acc is not None and acc + drop_tolerance < baseline_acc:
    raise SystemExit(
        f"RL smoke failed: eval acc@0.99={acc:.4f} dropped more than "
        f"{drop_tolerance:.4f} below baseline {baseline_acc:.4f}"
    )
if baseline_reward is not None and after + 1e-8 < baseline_reward:
    raise SystemExit(
        f"RL smoke failed: eval mean reward={after:.4f} is below baseline "
        f"mean reward {baseline_reward:.4f}"
    )
PY

echo "RL CPU smoke test passed. Outputs: $OUT_DIR"
