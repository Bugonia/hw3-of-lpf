#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
MODEL_DIR="${MODEL_DIR:-${PUBLIC_ROOT}/models/Qwen3-VL-8B-Instruct}"
BEST_ADAPTER="${BEST_ADAPTER:-${PUBLIC_ROOT}/hw3-of-lpf-best/qwen3_vl_v4_targeted_20260621/adapter}"
SAMPLES="${SAMPLES:-${PUBLIC_ROOT}/hw3-of-lpf/data/sft_v2/samples_train.jsonl}"
DATA_ROOT="${DATA_ROOT:-$(dirname "$SAMPLES")}"
OUT_DIR="${OUT_DIR:-outputs/rl_dpo_preflight}"

cd "$PROJECT_ROOT"

python3 - <<'PY'
from importlib import metadata
required = ["torch", "transformers", "peft", "accelerate", "numpy", "pillow"]
for name in required:
    try:
        print(f"{name}: {metadata.version(name)}")
    except Exception as exc:
        raise SystemExit(f"missing required package {name}: {exc}")
PY

for path in "$MODEL_DIR" "$BEST_ADAPTER" "$SAMPLES" "$DATA_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path missing: $path" >&2
    exit 1
  fi
done

python3 -m py_compile \
  scripts/generate_dpo_preference_data.py \
  scripts/train_qwen3_vl_dpo_lora.py \
  scripts/merge_qwen3_vl_lora.py \
  eval.py

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

python3 scripts/generate_dpo_preference_data.py \
  --samples "$SAMPLES" \
  --data-root "$DATA_ROOT" \
  --out "$OUT_DIR/preferences.jsonl" \
  --max-samples 4

python3 scripts/train_qwen3_vl_dpo_lora.py \
  --model-name-or-path "$MODEL_DIR" \
  --adapter-name-or-path "$BEST_ADAPTER" \
  --train-jsonl "$OUT_DIR/preferences.jsonl" \
  --output-dir "$OUT_DIR/adapter" \
  --max-samples 2 \
  --dry-run-batch

echo "RL GPU preflight passed. Generated dry-run data under $OUT_DIR"
