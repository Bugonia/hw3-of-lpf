#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/generative-large-model/public/hw3-of-lpf}"
OUT_DIR="${OUT_DIR:-data/stage3_balanced}"
SEED="${SEED:-20260621}"

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

  bash scripts/generate_stage3_balanced_data.sh
EOF
  exit 1
fi

# Counts are based on Stage-2 dev analysis:
# - focus remaining low-accuracy templates;
# - include templates with Stage-1 -> Stage-2 regressions to recover stability;
# - keep a small rehearsal set for high-performing families.
TEMPLATE_SAMPLES="${TEMPLATE_SAMPLES:-\
L1_sin=80,\
L1_cos=100,\
L1_exp_grow=260,\
L1_exp_decay=260,\
L1_sqrt=80,\
L1_poly=650,\
L2_gaussian=100,\
L2_sin_plus_linear=280,\
L2_sin_full=650,\
L2_sin_cos=500,\
L2_log=280,\
L2_exp_offset=160,\
L2_cos_full=320,\
L3_chirp=100,\
L3_sqrt_sin=360,\
L3_damped_osc=560,\
L3_gauss_sin=300,\
L3_beat=900,\
L3_growing_osc=360,\
L4_log_sin=560,\
L4_three_terms=360,\
L4_exp_chirp=500,\
L4_sqrt_cos_sq=760,\
L4_sin_of_exp=220,\
L5_sqrt_chirp_poly=760,\
L5_tanh_nested=700,\
L5_exp_sin_sq=900,\
L5_fm_signal=760,\
L5_log_sin_sq_cos=520\
}"

python3 scripts/generate_stage1_data.py \
  --out "$OUT_DIR" \
  --template-samples "$TEMPLATE_SAMPLES" \
  --seed "$SEED" \
  --min-points "${MIN_POINTS:-8}" \
  --max-points "${MAX_POINTS:-24}" \
  --n-test-points "${N_TEST_POINTS:-50}" \
  --val-ratio "${VAL_RATIO:-0.05}" \
  --overwrite

du -sh "$OUT_DIR"
