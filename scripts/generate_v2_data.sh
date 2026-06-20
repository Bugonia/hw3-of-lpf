#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/generative-large-model/public/hw3-of-lpf}"
OUT_DIR="${OUT_DIR:-data/sft_v2}"
SEED="${SEED:-20260622}"

cd "$PROJECT_ROOT"

if ! python3 - <<'PY' >/dev/null 2>&1
import matplotlib  # noqa: F401
import numpy  # noqa: F401
PY
then
  cat >&2 <<'EOF'
Missing data-generation dependencies in the active Python environment.

Install once in the shared hw3 environment from the online instance:

  source /inspire/hdd/project/generative-large-model/public/envs/hw3/bin/activate
  python -m pip install -U matplotlib numpy pillow
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
  --tool-only-ratio "${TOOL_ONLY_RATIO:-0.05}"
  --num-hard-negatives "${NUM_HARD_NEGATIVES:-4}"
  --num-candidate-families "${NUM_CANDIDATE_FAMILIES:-3}"
  --max-family-param-guesses "${MAX_FAMILY_PARAM_GUESSES:-512}"
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
