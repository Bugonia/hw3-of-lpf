# RL Post-Training Plan And CPU Smoke Workflow

## Goal

Use the current best SFT model as the starting point:

```text
/inspire/hdd/project/generative-large-model/public/hw3-of-lpf-best/current_best/model
```

The current best dev score is:

```text
acc@0.99 = 68.00%
acc@0.95 = 68.67%
acc@0.90 = 70.00%
acc@0.80 = 72.33%
null = 0
median R2 = 1.0
```

The remaining errors are mostly close-family or close-parameter failures:

- frequency off by one in beat/FM signals,
- Gaussian or damping coefficient errors,
- extra hallucinated terms,
- weak polynomial handling,
- confusion among `sqrt(cos(x**2))`, `log(abs(sin(...)))`, and `exp(sin(...)**2)` families.

## Recommended RL Strategy

Do not start with full PPO. The task has a deterministic verifier, so the most
stable route is verifier-guided preference and group policy optimization:

1. Generate targeted prompts from weak templates:
   `L1_poly`, `L2_sin_cos`, `L3_beat`, `L3_gauss_sin`,
   `L4_log_sin`, `L4_sqrt_cos_sq`, `L5_fm_signal`,
   `L5_sqrt_chirp_poly`, and `L5_exp_sin_sq`.
2. Sample multiple expressions from the current best model on GPU.
3. Parse tool calls and score each expression on hidden verification points.
4. Build preference pairs:
   - chosen: valid expression with `R2 >= 0.99`, or the highest-reward response;
   - rejected: valid but lower-reward close mistake, parse failure, or expression
     with extra terms.
5. Train DPO/IPO first, with rehearsal from already-solved templates.
6. If DPO improves dev, run GRPO with 4-8 samples per prompt and verifier reward.

The reward should combine format validity and numerical correctness:

```text
parse/eval failure: -1.0
valid expression: small positive format bonus
R2 clipped to [-1, 1]
R2 >= 0.80: near bonus
R2 >= 0.95: larger near bonus
R2 >= 0.99: pass bonus
small length/extra-term penalty when needed
```

## CPU Baseline-Aware Closed Loop

The CPU workflow in this repo validates the RL plumbing without loading Qwen.
It trains a tiny policy over candidate expressions:

- state: verifier features computed from reference points;
- action: choose one candidate expression;
- reward: hidden-test R2 shaped by `scripts/rl_expr_utils.py`;
- optimizer: REINFORCE;
- evaluation: greedy selected expression on held-out rows.

The default run is baseline-aware. It reads current-best dev predictions from:

```text
/inspire/hdd/project/generative-large-model/public/hw3-of-lpf-best/qwen3_vl_v4_targeted_20260621/eval_outputs/eval_results_dev.jsonl
```

For each dev sample, the candidate set includes:

- the current-best SFT prediction as the baseline candidate,
- the ground-truth expression, used here only to prove the CPU reward/optimization
  plumbing can select a high-reward candidate,
- numeric mutations of the ground truth and baseline expression,
- generic distractor expressions.

This is intentionally small and CPU-only. It does not claim to fine-tune Qwen.
It proves that data generation, verifier reward, RL optimization,
checkpointing, evaluation, and no-regression gating are connected before moving
the same reward/data interface to GPU Qwen sampling.

Run:

```bash
bash scripts/setup_rl_cpu_env.sh
bash scripts/run_rl_cpu_smoke.sh
```

Useful overrides:

```bash
OUT_DIR=outputs/rl_cpu_smoke \
MAX_SAMPLES=300 \
MAX_CANDIDATES=12 \
EPOCHS=80 \
DROP_TOLERANCE=0.02 \
bash scripts/run_rl_cpu_smoke.sh
```

Outputs:

```text
outputs/rl_cpu_smoke/data/manifest.json
outputs/rl_cpu_smoke/data/rl_train.jsonl
outputs/rl_cpu_smoke/data/rl_eval.jsonl
outputs/rl_cpu_smoke/policy/candidate_policy.pt
outputs/rl_cpu_smoke/policy/train_summary.json
outputs/rl_cpu_smoke/eval/eval_summary.json
outputs/rl_cpu_smoke/eval/eval_predictions.jsonl
```

The smoke test is considered valid only if:

- eval mean reward improves after RL training;
- eval `acc@0.99 >= 0.50`;
- when baseline results are available, RL eval `acc@0.99` is not more than
  `DROP_TOLERANCE` below the baseline eval `acc@0.99`;
- when baseline results are available, RL eval mean reward is not below baseline
  eval mean reward;
- the command exits successfully.

## Verified CPU Run

The following commands were run successfully on CPU:

```bash
bash scripts/setup_rl_cpu_env.sh
python3 -m py_compile scripts/rl_expr_utils.py scripts/generate_rl_candidate_data.py scripts/train_rl_candidate_policy.py scripts/evaluate_rl_candidate_policy.py
bash -n scripts/run_rl_cpu_smoke.sh
bash scripts/run_rl_cpu_smoke.sh
```

Environment:

```text
python: 3.12.13
numpy: 2.2.6
torch: 2.11.0+cu130
```

Final CPU smoke manifest:

```text
num_total: 300
num_train: 225
num_eval: 75
baseline coverage: 300/300
baseline_total acc@0.99: 0.6800
baseline_eval acc@0.99: 0.6533
baseline_eval mean_reward: 1.6234
```

Final RL result on the held-out CPU eval split:

```text
before_eval_mean_reward: -0.9334
after_eval_mean_reward: 2.5000
eval_acc@0.99: 1.0000
eval_mean_reward: 2.5000
```

No-drop gate:

```text
baseline_eval acc@0.99 = 0.6533
RL eval acc@0.99       = 1.0000
DROP_TOLERANCE         = 0.0200
status                 = passed
```

Artifacts from the verified run:

```text
outputs/rl_cpu_smoke/data/manifest.json
outputs/rl_cpu_smoke/data/rl_candidates.jsonl
outputs/rl_cpu_smoke/data/rl_train.jsonl
outputs/rl_cpu_smoke/data/rl_eval.jsonl
outputs/rl_cpu_smoke/policy/candidate_policy.pt
outputs/rl_cpu_smoke/policy/train_summary.json
outputs/rl_cpu_smoke/eval/eval_summary.json
outputs/rl_cpu_smoke/eval/eval_predictions.jsonl
```

## GPU Migration

On GPU, replace the synthetic candidate list with sampled Qwen outputs:

1. Keep the verifier functions in `scripts/rl_expr_utils.py`.
2. For each prompt, sample `K=8-32` responses from current best model.
3. Extract expressions with the same tool-call parser logic as `eval.py`.
4. Score each expression with hidden verification points.
5. Convert scored responses to DPO/IPO pairs or GRPO groups.
6. Train from current best LoRA or merged model, keeping rehearsal data mixed in.

The current best model was obtained with language-side LoRA only, and a vision
probe did not beat it. Start GPU RL from language-side adapters unless new
evidence shows the visual side is the bottleneck.
