# Technical Route

This project should be treated as a small post-training system for multimodal
symbolic regression, not as a fixed 29-class formula classifier.

## Core Understanding

The model receives a curve image, function hints, and reference points, then
must call `submit_expression(expression)` with a valid numpy expression.

The image and reference points play different roles:

- The image is best used for coarse structure: periodicity, symmetry,
  monotonicity, rough zero-crossing count, local extrema, envelopes, trends,
  and whether frequency changes over the visible range.
- Reference points are the decisive evidence for parameters. They should be
  used to test candidate expressions and reject close hard negatives such as
  `sin(3*x)` vs `sin(2*x)`, `cos(4*x)` vs `cos(3*x)`, and
  `exp(-x**2)` vs `exp(-0.5*x**2)`.

The 29 synthetic families are a strong in-distribution curriculum, but they are
not assumed to be the full hidden-test universe. The training goal is to teach a
workflow:

```text
read visual structure -> shortlist candidate families -> guess parameters ->
evaluate reference-point errors -> switch family/parameters if needed ->
submit only after the error is tiny
```

## Data Strategy

### V2: Current Clean Baseline

`scripts/generate_v2_data.sh` generates `data/sft_v2` from the 29 core
families. It includes:

- dev-like matplotlib images,
- function hints with distractors,
- Chebyshev reference points,
- hard negatives from nearby parameters,
- wrong-family candidates,
- mixed assistant targets: mostly point-check reasoning, plus a small tool-only
  slice.

The generator now also records deterministic `visual_features` computed from
dense sampling of the true expression. These features are used as training
scaffolding in the assistant reasoning. They are not extra input at eval time;
the model must learn to infer them from the image.

Point-check targets should force the model to substitute candidate expressions
into several reference points and compare predicted y-values against target
y-values. The accepted expression must be selected by the smallest verified
`max_abs_error`, not by candidate order. In particular, the correct candidate
inside the true family should be shuffled among hard negatives so the model
cannot learn "guess 1 is usually correct."

### V3: Next Data Upgrade

The next generator should extend beyond fixed templates:

1. Keep the 29 families as the core curriculum.
2. Add grammar-generated expressions from primitive numpy functions:
   `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`, `tanh`, powers, sums,
   products, offsets, envelopes, and nested phases.
3. Build hard negatives by:
   - changing one sensitive parameter,
   - swapping close families,
   - adding/removing trend terms,
   - changing envelope width or damping rate,
   - confusing linear phase with nonlinear phase.
4. Use the deterministic verifier to compute max absolute error and R2.
5. Only keep reasoning traces whose final expression is verified.

## Visual Feature Extractor

The extractor should be deterministic and conservative. It should not try to
perfectly solve the task. It should produce approximate, high-value labels:

- symmetry: even-like, odd-like, weak, or unclear,
- zero-crossing count and rough locations,
- local extrema count,
- monotonicity,
- rough y-range,
- oscillatory vs non-oscillatory,
- amplitude trend or centered envelope,
- changing local frequency.

These labels are useful for SFT because they teach the model what to look for in
the plot. They should be phrased as approximate observations, not as infallible
facts.

## Teacher Model Policy

Open-source stronger models may be used for data construction or debugging, but
they should not be the source of correctness.

Recommended use:

- ask a teacher model to explain failure cases or propose candidate families;
- use deterministic visual features and reference-point errors as ground truth;
- run a verifier on every teacher-written sample;
- keep only samples with correct tool calls and faithful reasoning.

Do not use external closed APIs for training data.

## Training Strategy

Mainline SFT:

- Base: Qwen3-VL-8B-Instruct.
- LoRA rank: start with `r=32`, consider `r=64` if data quality improves.
- Vision-side LoRA: enabled by default, because the task depends on curve-image
  structure.
- Precision: bf16, no 4-bit when enough H100 memory is available.
- Data mix: mostly structured point-check reasoning, with some tool-only samples
  to preserve stable final calls.

After the clean SFT baseline:

1. Evaluate on dev.
2. Analyze failures by expression family, visual feature, R2 bucket, and null
   output.
3. Generate targeted repair data from the failure modes.
4. Use self-sampling plus verifier scoring to create rejection-SFT or preference
   data.
5. Keep all experiment outputs under `outputs/` and avoid committing generated
   data or checkpoints.

## Success Criteria

A good model should:

- use the image to rule out implausible families;
- use reference points to choose sensitive parameters;
- recover from a wrong candidate by switching family or parameter;
- produce valid tool calls reliably;
- generalize beyond the exact 29 templates when the curve structure is familiar.
