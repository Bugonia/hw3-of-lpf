#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/hdd/project/generative-large-model/public/hw3-of-lpf}"
OUT_DIR="${OUT_DIR:-data/stage2_targeted}"
SEED="${SEED:-20260620}"

cd "$PROJECT_ROOT"

# Counts are based on the first Stage-1 dev analysis:
# - keep every family represented;
# - heavily oversample templates with low acc@0.99 or catastrophic R2 failures.
TEMPLATE_SAMPLES="${TEMPLATE_SAMPLES:-\
L1_sin=120,\
L1_cos=120,\
L1_exp_grow=180,\
L1_exp_decay=160,\
L1_sqrt=120,\
L1_poly=160,\
L2_gaussian=240,\
L2_sin_plus_linear=450,\
L2_sin_full=600,\
L2_sin_cos=500,\
L2_log=240,\
L2_exp_offset=220,\
L2_cos_full=180,\
L3_chirp=220,\
L3_sqrt_sin=300,\
L3_damped_osc=180,\
L3_gauss_sin=360,\
L3_beat=700,\
L3_growing_osc=500,\
L4_log_sin=520,\
L4_three_terms=520,\
L4_exp_chirp=240,\
L4_sqrt_cos_sq=650,\
L4_sin_of_exp=700,\
L5_sqrt_chirp_poly=560,\
L5_tanh_nested=650,\
L5_exp_sin_sq=700,\
L5_fm_signal=560,\
L5_log_sin_sq_cos=760\
}"

python3 scripts/generate_stage1_data.py \
  --out "$OUT_DIR" \
  --template-samples "$TEMPLATE_SAMPLES" \
  --seed "$SEED" \
  --min-points "${MIN_POINTS:-6}" \
  --max-points "${MAX_POINTS:-22}" \
  --n-test-points "${N_TEST_POINTS:-50}" \
  --val-ratio "${VAL_RATIO:-0.05}" \
  --overwrite

du -sh "$OUT_DIR"
