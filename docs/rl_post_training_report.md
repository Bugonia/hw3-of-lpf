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

The repo now includes a real GPU DPO/RL training path. It is not the CPU smoke
candidate-policy test. It loads Qwen3-VL, starts from the current-best LoRA
adapter, trains on verifier-built chosen/rejected preference data, then can
merge and evaluate the resulting adapter.

### Offline Environment

On the GPU machine, use the project-local activation wrapper:

```bash
cd /inspire/hdd/project/generative-large-model/public/ywy/hw3-of-lpf
source envs/rl_gpu/activate.sh
```

This activates the shared offline environment:

```text
/inspire/hdd/project/generative-large-model/public/envs/hw3
```

and sets offline/cache flags such as `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `PIP_NO_INDEX=1`, and audio-disable flags needed for
fast PEFT import.

Expected package versions in the verified environment:

```text
torch: 2.11.0
transformers: 5.13.0.dev0
peft: 0.19.1
accelerate: 1.14.0
numpy: 2.4.6
pillow: 12.2.0
vllm: 0.23.0
bitsandbytes: 0.49.2
```

### Preflight On A Fresh Shared Machine

Run this before starting a long GPU job:

```bash
source envs/rl_gpu/activate.sh
bash scripts/preflight_rl_gpu.sh
```

Verified output on this machine:

```text
Activated offline RL env: /inspire/hdd/project/generative-large-model/public/envs/hw3
torch: 2.11.0
transformers: 5.13.0.dev0
peft: 0.19.1
accelerate: 1.14.0
numpy: 2.4.6
pillow: 12.2.0
num_rows: 4
dry run ok
RL GPU preflight passed.
```

What this proves:

- the shared env is present and importable offline;
- the base model, best adapter, synthetic SFT data, and images are present;
- DPO preference JSONL can be generated;
- Qwen3-VL `AutoProcessor.apply_chat_template` can encode image+text
  chosen/rejected batches;
- the DPO trainer entrypoint can start through batch construction.

### Real GPU DPO/RL Training

Default paths:

```text
base model:
  /inspire/hdd/project/generative-large-model/public/models/Qwen3-VL-8B-Instruct
starting adapter:
  /inspire/hdd/project/generative-large-model/public/hw3-of-lpf-best/qwen3_vl_v4_targeted_20260621/adapter
training samples:
  /inspire/hdd/project/generative-large-model/public/hw3-of-lpf/data/sft_v2/samples_train.jsonl
output:
  outputs/rl_dpo/
```

Start a conservative first GPU run:

```bash
source envs/rl_gpu/activate.sh

CUDA_VISIBLE_DEVICES=0 bash scripts/run_rl_dpo_safe_train.sh
```

This safe wrapper is the recommended entrypoint after the first aggressive
100-step run regressed dev from 68.0% to 38.7%. The regression was likely caused
by a combination of:

- rejected samples being too easy (`generic` or obviously invalid alternatives),
  so the DPO objective did not teach subtle parameter discrimination;
- effective batch size only 8, which is noisy for preference optimization;
- learning rate/beta/update count too strong for a model already at 68%;
- all chosen samples being synthetic ground truth, which can pull the model away
  from the current-best response distribution.

Safe defaults:

```text
MAX_PREF_SAMPLES=2000
REJECTION_MODE=hardest
MAX_STEPS=30
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=5e-7
DPO_BETA=0.03
SFT_LOSS_COEF=0.03
WARMUP_RATIO=0.1
```

On the 140G single-GPU machine, `PER_DEVICE_TRAIN_BATCH_SIZE=4` with
`GRADIENT_ACCUMULATION_STEPS=8` gives effective batch 32 while reducing
accumulation overhead. If memory is still clearly below capacity, try
`PER_DEVICE_TRAIN_BATCH_SIZE=8` and `GRADIENT_ACCUMULATION_STEPS=4`, keeping
the effective batch unchanged.

The first safe run reported by the GPU machine reached:

```text
acc@0.99 = 67.0%
acc@0.95 = 67.7%
acc@0.90 = 69.0%
acc@0.80 = 72.0%
median R2 = 1.0
```

This is not an improvement over the current-best SFT checkpoint
(`acc@0.99 = 68.0%`). Do not simply train longer from that run.

### Guarded Long Training

For longer training, use the guarded script. It trains in phases, evaluates
after every phase, accepts a phase only if it improves dev `acc@0.99`, and stops
if the score drops below the SFT baseline.

Use the global-storage wrapper on machines where the project filesystem is full:

```bash
source envs/rl_gpu/activate.sh

CUDA_VISIBLE_DEVICES=0 \
RUN_NAME=guarded_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_rl_dpo_global_guarded_train.sh
```

This stores all heavy artifacts under:

```text
/inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/<RUN_NAME>/
```

including preference JSONL, phase adapters, merged models, eval logs, and eval
summaries.

Equivalent manual guarded command:

```bash
source envs/rl_gpu/activate.sh

CUDA_VISIBLE_DEVICES=0 \
RUN_DIR=/inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/manual_guarded \
TMP_MERGE_ROOT=/inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/manual_guarded/merged_models \
EVAL_OUTPUT_ROOT=/inspire/hdd/global_user/yuwenye-253108120175/hw3_rl_runs/manual_guarded/eval_outputs \
PHASES=8 \
PHASE_STEPS=10 \
PER_DEVICE_TRAIN_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=4 \
LEARNING_RATE=2e-7 \
DPO_BETA=0.02 \
SFT_LOSS_COEF=0.05 \
bash scripts/run_rl_dpo_guarded_long_train.sh
```

Defaults:

```text
BASELINE_ACC=0.68
MIN_KEEP_ACC=0.68
MIN_IMPROVEMENT=0.001
PHASES=8
PHASE_STEPS=10
MAX_PREF_SAMPLES=3000
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-7
DPO_BETA=0.02
SFT_LOSS_COEF=0.05
REJECTION_MODE=hardest
```

Outputs:

```text
outputs/rl_dpo_guarded/phase_*/adapter/
outputs/rl_dpo_guarded/phase_*/eval_logs/
outputs/rl_dpo_guarded/guarded_status.jsonl
eval_outputs/qwen3_vl_dpo_guarded_phase_*_merged/eval_summary_dev.json
```

Use the adapter from the last `"decision": "accept"` row in
`outputs/rl_dpo_guarded/guarded_status.jsonl`. If every phase is rejected or the
run stops early, keep the original current-best SFT checkpoint.

Full one-epoch run over the generated preference data:

```bash
source envs/rl_gpu/activate.sh

CUDA_VISIBLE_DEVICES=0 \
NUM_TRAIN_EPOCHS=1 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=8 \
LEARNING_RATE=5e-6 \
DPO_BETA=0.1 \
bash scripts/run_rl_dpo_train.sh
```

The training wrapper does two things:

1. `scripts/generate_dpo_preference_data.py`
   - chosen response: verified ground-truth expression with short tool-call
     answer;
   - rejected response: current-best baseline mistake when available, or
     verifier-scored numeric/generic hard negative;
   - output: `outputs/rl_dpo/data/preferences.jsonl`.
2. `scripts/train_qwen3_vl_dpo_lora.py`
   - loads Qwen3-VL base model;
   - loads current-best LoRA as trainable adapter;
   - computes chosen/rejected assistant log-probabilities;
   - optimizes `-logsigmoid(beta * (logp_chosen - logp_rejected))`;
   - saves adapter to `outputs/rl_dpo/qwen3_vl_dpo_lora`.

This is reference-free DPO-style preference optimization. It avoids a second
8B reference model so it can start on a single GPU. The current-best adapter is
the initialization, and the verifier creates the preference signal.

### Merge And Evaluate

After training:

```bash
source envs/rl_gpu/activate.sh

CUDA_VISIBLE_DEVICES=0 \
ADAPTER_DIR=outputs/rl_dpo/qwen3_vl_dpo_lora \
MERGED_DIR=outputs/rl_dpo/qwen3_vl_dpo_merged \
bash scripts/run_rl_dpo_eval.sh
```

This runs:

- `scripts/merge_qwen3_vl_lora.py`,
- then official `eval.py --split dev --enforce-eager`.

Outputs:

```text
outputs/rl_dpo/qwen3_vl_dpo_lora/
outputs/rl_dpo/qwen3_vl_dpo_merged/
eval_outputs/qwen3_vl_dpo_merged/eval_results_dev.jsonl
eval_outputs/qwen3_vl_dpo_merged/eval_summary_dev.json
```

Compare against current best:

```text
current best dev acc@0.99 = 0.6800
```

If the RL/DPO eval drops significantly, do not submit it. Lower the learning
rate, reduce steps, add more rehearsal/easy rows, or switch to smaller targeted
template subsets.

### Later GRPO Variant

For a true online GRPO stage, replace the generated rejected candidates with
sampled Qwen outputs:

1. Keep the verifier functions in `scripts/rl_expr_utils.py`.
2. For each prompt, sample `K=8-32` responses from current best model.
3. Extract expressions with the same tool-call parser logic as `eval.py`.
4. Score each expression with hidden verification points.
5. Convert scored responses to DPO/IPO pairs or GRPO groups.
6. Train from current best LoRA or merged model, keeping rehearsal data mixed in.

The current best model was obtained with language-side LoRA only, and a vision
probe did not beat it. Start GPU RL from language-side adapters unless new
evidence shows the visual side is the bottleneck.
