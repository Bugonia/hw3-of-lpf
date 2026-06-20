# Qwen3-VL LoRA Training

This is the first SFT baseline for the stage-1 synthetic data.

## 1. Update the Public Repo

```bash
cd /inspire/hdd/project/generative-large-model/public/hw3-of-lpf
git pull --ff-only origin main
```

## 2. Create the Shared `hw3` Environment

The online and offline instances can both access the project public directory, so
create the environment directly there from the online instance:

```bash
bash scripts/setup_shared_hw3_env.sh
```

This creates:

```text
/inspire/hdd/project/generative-large-model/public/envs/hw3
```

Activate the same environment from either instance:

```bash
source /inspire/hdd/project/generative-large-model/public/envs/hw3/bin/activate
```

If the base image already has a working CUDA PyTorch and you do not want to
install PyTorch into the shared environment, use `INSTALL_TORCH=0`.

## 3. Install Training Dependencies Only

The Qwen3-VL model card recommends a recent Transformers build. Install once on
the server image or use `INSTALL_DEPS=1 INSTALL_ONLY=1` in the run script.

```bash
INSTALL_DEPS=1 INSTALL_ONLY=1 bash scripts/run_stage1_lora.sh
```

## 4. Data Collation Dry Run

After the base model is available, check the processor/chat-template path without
loading model weights:

```bash
DRY_RUN_BATCH=1 bash scripts/run_stage1_lora.sh
```

`DRY_RUN_BATCH=1` builds one multimodal batch and exits before loading the model.

## 5. Smoke Train

After the base model is available at
`/inspire/hdd/project/generative-large-model/public/models/Qwen3-VL-8B-Instruct`,
run a one-step training smoke test:

```bash
MAX_STEPS=1 MAX_TRAIN_SAMPLES=16 MAX_EVAL_SAMPLES=4 bash scripts/run_stage1_lora.sh
```

When `NPROC_PER_NODE=1`, the run script narrows a multi-GPU
`CUDA_VISIBLE_DEVICES` value to the first visible GPU. This avoids PyTorch
`DataParallel`, which is brittle for Qwen3-VL's vision tower. Multi-GPU training
uses `torchrun` by setting `NPROC_PER_NODE`.

## 6. Stage-1 LoRA Train

```bash
bash scripts/run_stage1_lora.sh
```

Default output:

```text
/inspire/hdd/project/generative-large-model/public/hw3-of-lpf/outputs/qwen3_vl_stage1_lora
```

The run script defaults to 4-bit LoRA (`LOAD_IN_4BIT=1`) for single-GPU memory
safety. Override any hyperparameter with environment variables, for example:

```bash
NUM_TRAIN_EPOCHS=2 LEARNING_RATE=1e-4 LORA_R=32 bash scripts/run_stage1_lora.sh
```

The default attention backend is PyTorch SDPA so the environment does not need
FlashAttention2. If `flash-attn` is installed, you can opt in with:

```bash
ATTN_IMPLEMENTATION=flash_attention_2 bash scripts/run_stage1_lora.sh
```

## 7. Merge for Official Eval

The official `eval.py` loads a normal model directory, so merge the adapter:

```bash
bash scripts/merge_stage1_lora.sh
```

Default merged output:

```text
/inspire/hdd/project/generative-large-model/public/hw3-of-lpf/outputs/qwen3_vl_stage1_merged
```

Then run dev evaluation:

```bash
python eval.py /inspire/hdd/project/generative-large-model/public/hw3-of-lpf/outputs/qwen3_vl_stage1_merged \
  --split dev \
  --tp 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser hermes
```

## 8. Stage-2 Targeted Continuation

After Stage 1, the first dev run reached `acc@0.99=42.0%` with `null=0%`.
Errors concentrated in the templates with many distractors and nested functions
(`L5_log_sin_sq_cos`, `L3_beat`, `L5_exp_sin_sq`, `L4_sin_of_exp`,
`L4_sqrt_cos_sq`, and related families). Generate a targeted set:

```bash
python -m pip install -U matplotlib numpy pillow
bash scripts/generate_stage2_targeted_data.sh
```

Continue training from the Stage-1 adapter on four H100 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
LOAD_IN_4BIT=0 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
NUM_TRAIN_EPOCHS=1 \
LEARNING_RATE=5e-5 \
bash scripts/run_stage2_lora.sh
```

Merge and evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/merge_stage2_lora.sh

CUDA_VISIBLE_DEVICES=0 python eval.py \
  /inspire/hdd/project/generative-large-model/public/hw3-of-lpf/outputs/qwen3_vl_stage2_merged \
  --split dev \
  --tp 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser hermes \
  --enforce-eager
```

## 9. Output Directory Convention

All project results should live under the repository directory:

```text
/inspire/hdd/project/generative-large-model/public/hw3-of-lpf
```

Training adapters and merged models default to:

```text
/inspire/hdd/project/generative-large-model/public/hw3-of-lpf/outputs/
```

Evaluation summaries remain in the repository-local `eval_outputs/` directory.
If older runs were written to `/inspire/hdd/project/generative-large-model/public/outputs`,
copy them into the project directory with:

```bash
bash scripts/migrate_outputs_to_project.sh
```

## 10. Stage-3 Conservative Continuation

Stage 2 improved dev `acc@0.99` from `42.0%` to `58.3%`, fixing 62 samples
while regressing 13. Stage 3 targets the remaining low-score templates
(`L3_beat`, `L5_exp_sin_sq`, `L1_poly`, `L5_sqrt_chirp_poly`,
`L5_fm_signal`, `L4_sqrt_cos_sq`) and rehearses the regression-prone templates
(`L3_damped_osc`, `L4_exp_chirp`, `L4_log_sin`, `L2_sin_full`).

Generate data:

```bash
python -m pip install -U matplotlib numpy pillow
bash scripts/generate_stage3_balanced_data.sh
```

Continue from Stage 2 with a lower learning rate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
LOAD_IN_4BIT=0 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
NUM_TRAIN_EPOCHS=1 \
LEARNING_RATE=2e-5 \
bash scripts/run_stage3_lora.sh
```

Stage 3 defaults to `SAVE_STRATEGY=no`, so it skips intermediate checkpoints and
only writes the final adapter at the end. This avoids filling the shared project
quota during continuation runs.

Merge and evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/merge_stage3_lora.sh

CUDA_VISIBLE_DEVICES=0 python eval.py \
  /inspire/hdd/project/generative-large-model/public/hw3-of-lpf/outputs/qwen3_vl_stage3_merged \
  --split dev \
  --tp 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser hermes \
  --enforce-eager
```
