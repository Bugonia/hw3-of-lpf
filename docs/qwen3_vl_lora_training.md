# Qwen3-VL LoRA Training

This is the first SFT baseline for the stage-1 synthetic data.

## 1. Update the Public Repo

```bash
cd /inspire/hdd/project/generative-large-model/public/hw3-of-lpf
git pull --ff-only origin main
```

## 2. Install Training Dependencies

The Qwen3-VL model card recommends a recent Transformers build. Install once on
the server image or use `INSTALL_DEPS=1 INSTALL_ONLY=1` in the run script.

```bash
INSTALL_DEPS=1 INSTALL_ONLY=1 bash scripts/run_stage1_lora.sh
```

## 3. Data Collation Dry Run

After the base model is available, check the processor/chat-template path without
loading model weights:

```bash
DRY_RUN_BATCH=1 bash scripts/run_stage1_lora.sh
```

`DRY_RUN_BATCH=1` builds one multimodal batch and exits before loading the model.

## 4. Smoke Train

After the base model is available at
`/inspire/hdd/project/generative-large-model/public/models/Qwen3-VL-8B-Instruct`,
run a one-step training smoke test:

```bash
MAX_STEPS=1 MAX_TRAIN_SAMPLES=16 MAX_EVAL_SAMPLES=4 bash scripts/run_stage1_lora.sh
```

## 5. Stage-1 LoRA Train

```bash
bash scripts/run_stage1_lora.sh
```

Default output:

```text
/inspire/hdd/project/generative-large-model/public/outputs/qwen3_vl_stage1_lora
```

The run script defaults to 4-bit LoRA (`LOAD_IN_4BIT=1`) for single-GPU memory
safety. Override any hyperparameter with environment variables, for example:

```bash
NUM_TRAIN_EPOCHS=2 LEARNING_RATE=1e-4 LORA_R=32 bash scripts/run_stage1_lora.sh
```

## 6. Merge for Official Eval

The official `eval.py` loads a normal model directory, so merge the adapter:

```bash
bash scripts/merge_stage1_lora.sh
```

Default merged output:

```text
/inspire/hdd/project/generative-large-model/public/outputs/qwen3_vl_stage1_merged
```

Then run dev evaluation:

```bash
python eval.py /inspire/hdd/project/generative-large-model/public/outputs/qwen3_vl_stage1_merged \
  --split dev \
  --tp 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser hermes
```
