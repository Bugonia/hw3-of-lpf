# Stage 1 Data Generation

`scripts/generate_stage1_data.py` creates synthetic symbolic-regression samples
for the first baseline. It covers the 29 dev-style function families observed in
the released validation set and writes both task-format metadata and SFT-ready
messages.

## Quick Start

Generate a small smoke-test set:

```bash
python3 scripts/generate_stage1_data.py --out data/stage1_smoke --num-samples 29 --overwrite
```

Generate a stronger first SFT set:

```bash
python3 scripts/generate_stage1_data.py --out data/stage1_synth --samples-per-template 100 --seed 20260619 --overwrite
```

The output directory contains:

- `images/`: 590x390 matplotlib curve plots.
- `samples.jsonl`: dev-like task records with image path, function hints,
  reference points, expression, test points, and generation config.
- `sft_messages.jsonl`: records with multimodal user messages and an assistant
  `<tool_call>` answer.
- `samples_train.jsonl` / `samples_val.jsonl`: task-format split files.
- `sft_train.jsonl` / `sft_val.jsonl`: SFT-ready split files.
- `manifest.json`: generation settings and covered template ids.

By default, `--num-samples 2900` is distributed as evenly as possible across the
29 templates. Use `--samples-per-template N` when you want exact balance. The
default validation ratio is 5%.

For targeted follow-up runs, pass exact per-template counts:

```bash
python3 scripts/generate_stage1_data.py \
  --out data/stage2_targeted \
  --template-samples 'L1_sin=120,L3_beat=700,L5_log_sin_sq_cos=760' \
  --overwrite
```

The repository also includes `scripts/generate_stage2_targeted_data.sh`, which
uses the first Stage-1 dev analysis to oversample the weakest templates while
keeping all 29 families represented.

## Point-Check Reasoning Data

For later stages, the same generator can write assistant targets that include a
short hard-negative check before the final tool call:

```bash
python3 scripts/generate_stage1_data.py \
  --out data/stage5_reasoning \
  --template-samples 'L2_sin_full=700,L2_cos_full=520,L2_gaussian=520' \
  --assistant-style point_check \
  --num-hard-negatives 4 \
  --overwrite
```

In this mode the user prompt is unchanged. The assistant target first states the
candidate family suggested by the image and function hints, then tests nearby
parameter variants on the provided reference points, for example `sin(3*x)`
versus `sin(2*x)` or `exp(-1*x**2)` versus `exp(-0.5*x**2)`. The final answer is
still a `submit_expression` tool call.

Use `scripts/generate_stage5_reasoning_data.sh` for the current hard-negative
recipe. It focuses families where adjacent frequencies, Gaussian widths, phases,
offsets, and coefficients are easy to confuse.

## Covered Families

The generator includes:

- L1: `sin`, `cos`, growing/decaying `exp`, `sqrt`, polynomial.
- L2: Gaussian, shifted sin/cos, `sin + linear`, `sin + cos`, log, exp offset.
- L3: chirp, sqrt-sin, damped oscillation, Gaussian-sin, beat, growing oscillation.
- L4: log-sin, three-term mixtures, exp-chirp, sqrt-cos-squared, sin-of-exp.
- L5: sqrt-chirp-polynomial, tanh-nested, exp-sin-squared, FM signal,
  log-sin-squared-cos.

Function hints always contain the true high-level functions and add distractors
according to difficulty. Reference points are Chebyshev nodes, matching the
released dev samples closely enough for a first baseline.
