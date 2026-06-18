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
python3 scripts/generate_stage1_data.py --out data/stage1_synth --num-samples 50000 --seed 20260619 --overwrite
```

The output directory contains:

- `images/`: 590x390 matplotlib curve plots.
- `samples.jsonl`: dev-like task records with image path, function hints,
  reference points, expression, test points, and generation config.
- `sft_messages.jsonl`: records with multimodal user messages and an assistant
  `<tool_call>` answer.
- `manifest.json`: generation settings and covered template ids.

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
