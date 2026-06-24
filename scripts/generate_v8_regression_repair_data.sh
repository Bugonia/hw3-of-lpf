#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/inspire/qb-ilm2/project/generative-large-model/zhongxiaoqiu-253108120179/hw3-of-lpf}"
OUT_DIR="${OUT_DIR:-data/sft_v8_regression_repair}"
SEED="${SEED:-20260626}"

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

# v8 starts from the v7 adapter. It emphasizes the families that regressed in
# v7, while replaying the families that v7 fixed so the small repair run keeps
# those gains.
TEMPLATE_SAMPLES_DEFAULT="\
L2_exp_offset=520,\
L2_sin_plus_linear=480,\
L4_sin_of_exp=460,\
L1_cos=420,\
L5_exp_sin_sq=420,\
L2_sin_cos=360,\
L3_beat=320,\
L5_fm_signal=320,\
L4_exp_chirp=260,\
L2_gaussian=220,\
L1_poly=220,\
L3_gauss_sin=220,\
L5_sqrt_chirp_poly=220,\
L4_log_sin=180,\
L3_sqrt_sin=180,\
L4_three_terms=140,\
L2_sin_full=140,\
L2_cos_full=120,\
L1_exp_decay=120,\
L1_exp_grow=100,\
L1_sin=80,\
L1_sqrt=80,\
L2_log=100,\
L3_chirp=100,\
L3_damped_osc=80,\
L3_growing_osc=80,\
L4_sqrt_cos_sq=100,\
L5_log_sin_sq_cos=100,\
L5_tanh_nested=120"

GEN_ARGS=(
  --out "$OUT_DIR"
  --seed "$SEED"
  --min-points "${MIN_POINTS:-8}"
  --max-points "${MAX_POINTS:-24}"
  --n-test-points "${N_TEST_POINTS:-50}"
  --val-ratio "${VAL_RATIO:-0.04}"
  --assistant-style "${ASSISTANT_STYLE:-mixed}"
  --mixed-reasoning-style "${MIXED_REASONING_STYLE:-short_check}"
  --tool-only-ratio "${TOOL_ONLY_RATIO:-0.65}"
  --num-hard-negatives "${NUM_HARD_NEGATIVES:-4}"
  --num-candidate-families "${NUM_CANDIDATE_FAMILIES:-1}"
  --max-family-param-guesses "${MAX_FAMILY_PARAM_GUESSES:-128}"
  --accept-max-abs-error "${ACCEPT_MAX_ABS_ERROR:-1e-4}"
  --num-verification-points "${NUM_VERIFICATION_POINTS:-4}"
  --template-samples "${TEMPLATE_SAMPLES:-$TEMPLATE_SAMPLES_DEFAULT}"
  --overwrite
)

python3 scripts/generate_sft_data.py "${GEN_ARGS[@]}"

python3 - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
prompt = manifest["prompt_template"].strip() + "\n"
(out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
print(f"Prompt template: {out_dir / 'prompt.txt'}")
PY

du -sh "$OUT_DIR"
