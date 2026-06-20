# Clean Workflow

This repo keeps only the release data, generic data generation, generic LoRA
training, merge, and evaluation-analysis utilities.

The project route is documented in `docs/technical_route.md`. In short: the
image is used to infer coarse visual structure, while reference points are used
to verify parameters and reject candidate expressions.

## Generate Data

```bash
bash scripts/generate_v2_data.sh
```

Defaults:

- output: `data/sft_v2`
- 300 samples per template across the 29 synthetic families
- mixed assistant targets: 20% direct tool-call, 80% multi-family point-check
- hard negatives and reference-point verification enabled
- deterministic visual-feature labels from dense samples of the true expression

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
LOAD_IN_4BIT=0 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
NUM_TRAIN_EPOCHS=1 \
LEARNING_RATE=1e-5 \
SAVE_STRATEGY=no \
bash scripts/run_lora_sft.sh
```

Defaults:

- base model: `/inspire/hdd/project/generative-large-model/public/models/Qwen3-VL-8B-Instruct`
- output adapter: `outputs/qwen3_vl_sft_lora`
- LoRA rank: `32`
- LoRA alpha: `64`
- vision-side LoRA: unfrozen by default

## Merge And Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/merge_lora.sh

CUDA_VISIBLE_DEVICES=0 python eval.py \
  /inspire/hdd/project/generative-large-model/public/hw3-of-lpf/outputs/qwen3_vl_sft_merged \
  --split dev \
  --tp 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser hermes \
  --enforce-eager
```

## Analyze

```bash
python scripts/analyze_eval_results.py \
  eval_outputs/qwen3_vl_sft_merged/eval_results_dev.jsonl \
  --csv-out eval_outputs/qwen3_vl_sft_merged/worst_dev.csv
```

Use the analysis to decide the next data upgrade. The preferred next step is not
another ad hoc stage script, but a new versioned data generator that adds grammar
generalization, verified hard negatives, and failure-mode repair data.
