#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/generative-large-model/zhongxiaoqiu-253108120179/hw3-of-lpf}"
OUT_DIR="${OUT_DIR:-data/sft_v3_short}"
SEED="${SEED:-20260623}"

cd "$PROJECT_ROOT"

if ! python3 - <<'PY' >/dev/null 2>&1
import matplotlib  # noqa: F401
import numpy  # noqa: F401
PY
then
  cat >&2 <<'EOF'
Missing data-generation dependencies in the active Python environment.

Use the existing shared hw3 environment; do not install from the offline server.
EOF
  exit 1
fi

GEN_ARGS=(
  --out "$OUT_DIR"
  --seed "$SEED"
  --min-points "${MIN_POINTS:-8}"
  --max-points "${MAX_POINTS:-24}"
  --n-test-points "${N_TEST_POINTS:-50}"
  --val-ratio "${VAL_RATIO:-0.04}"
  --assistant-style "${ASSISTANT_STYLE:-mixed}"
  --mixed-reasoning-style "${MIXED_REASONING_STYLE:-short_check}"
  --tool-only-ratio "${TOOL_ONLY_RATIO:-0.40}"
  --num-hard-negatives "${NUM_HARD_NEGATIVES:-4}"
  --num-candidate-families "${NUM_CANDIDATE_FAMILIES:-1}"
  --max-family-param-guesses "${MAX_FAMILY_PARAM_GUESSES:-128}"
  --accept-max-abs-error "${ACCEPT_MAX_ABS_ERROR:-1e-4}"
  --num-verification-points "${NUM_VERIFICATION_POINTS:-4}"
  --overwrite
)

if [[ -n "${TEMPLATE_SAMPLES:-}" ]]; then
  GEN_ARGS+=(--template-samples "$TEMPLATE_SAMPLES")
else
  GEN_ARGS+=(--samples-per-template "${SAMPLES_PER_TEMPLATE:-300}")
fi

python3 scripts/generate_sft_data.py "${GEN_ARGS[@]}"
du -sh "$OUT_DIR"
