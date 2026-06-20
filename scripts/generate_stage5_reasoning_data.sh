#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/generative-large-model/public/hw3-of-lpf}"
OUT_DIR="${OUT_DIR:-data/stage5_reasoning}"
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

Then rerun:

  bash scripts/generate_stage5_reasoning_data.sh
EOF
  exit 1
fi

# Stage 5 teaches the model to resolve visually plausible parameter confusions
# by checking hard-negative candidates against the provided reference points.
# It focuses families where nearby coefficients/frequencies often look similar:
# sin(3x) vs sin(2x), cos(4x) vs cos(3x), exp(-x**2) vs exp(-0.5*x**2),
# and coefficient choices such as 1/2/3.
TEMPLATE_SAMPLES="${TEMPLATE_SAMPLES:-\
L1_sin=180,\
L1_cos=220,\
L1_exp_grow=120,\
L1_exp_decay=160,\
L1_sqrt=80,\
L1_poly=320,\
L2_gaussian=520,\
L2_sin_plus_linear=240,\
L2_sin_full=700,\
L2_sin_cos=360,\
L2_log=160,\
L2_exp_offset=220,\
L2_cos_full=520,\
L3_chirp=160,\
L3_sqrt_sin=180,\
L3_damped_osc=320,\
L3_gauss_sin=440,\
L3_beat=420,\
L3_growing_osc=180,\
L4_log_sin=260,\
L4_three_terms=200,\
L4_exp_chirp=260,\
L4_sqrt_cos_sq=260,\
L4_sin_of_exp=560,\
L5_sqrt_chirp_poly=260,\
L5_tanh_nested=400,\
L5_exp_sin_sq=600,\
L5_fm_signal=320,\
L5_log_sin_sq_cos=260\
}"

python3 scripts/generate_stage1_data.py \
  --out "$OUT_DIR" \
  --template-samples "$TEMPLATE_SAMPLES" \
  --seed "$SEED" \
  --min-points "${MIN_POINTS:-10}" \
  --max-points "${MAX_POINTS:-24}" \
  --n-test-points "${N_TEST_POINTS:-50}" \
  --val-ratio "${VAL_RATIO:-0.04}" \
  --assistant-style point_check \
  --num-hard-negatives "${NUM_HARD_NEGATIVES:-4}" \
  --overwrite

du -sh "$OUT_DIR"
