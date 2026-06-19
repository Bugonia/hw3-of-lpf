#!/usr/bin/env python3
"""LoRA/QLoRA SFT for Qwen3-VL on the stage-1 symbolic-regression data."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


VISION_NAME_MARKERS = (
    "visual",
    "vision",
    "vit",
    "image",
    "patch_embed",
    "merger",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL-8B with LoRA SFT")
    parser.add_argument(
        "--model-name-or-path",
        default="/inspire/hdd/project/generative-large-model/public/models/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--train-jsonl", default="data/stage1_synth/sft_train.jsonl")
    parser.add_argument("--eval-jsonl", default="data/stage1_synth/sft_val.jsonl")
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--output-dir",
        default="/inspire/hdd/project/generative-large-model/public/outputs/qwen3_vl_stage1_lora",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated PEFT target module suffixes.",
    )
    parser.add_argument("--freeze-vision-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
    )
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument(
        "--dry-run-batch",
        action="store_true",
        help="Build one batch and exit without loading the model.",
    )
    return parser.parse_args()


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


class JsonlVLSftDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, data_root: str | Path | None, limit: int | None = None):
        self.jsonl_path = Path(jsonl_path)
        self.data_root = Path(data_root) if data_root else self.jsonl_path.parent
        self.rows = load_rows(self.jsonl_path, limit)
        if not self.rows:
            raise ValueError(f"No rows found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_messages(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        messages = copy.deepcopy(row["messages"])
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") != "image":
                    continue
                image = item.get("image")
                if not image or str(image).startswith(("http://", "https://", "file://")):
                    continue
                item["image"] = str((self.data_root / image).resolve())
        return messages

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        messages = self._resolve_messages(row)
        return {
            "id": row.get("id", str(idx)),
            "messages": messages,
            "prompt_messages": [messages[0]],
        }


@dataclass
class VLDataCollator:
    processor: Any
    max_length: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_messages = [feature["messages"] for feature in features]
        prompt_messages = [feature["prompt_messages"] for feature in features]

        full_inputs = self._encode(full_messages, add_generation_prompt=False)
        prompt_inputs = self._encode(prompt_messages, add_generation_prompt=True)

        labels = full_inputs["input_ids"].clone()
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, : min(int(prompt_len), labels.shape[1])] = -100
        labels[full_inputs["attention_mask"] == 0] = -100
        full_inputs["labels"] = labels
        return full_inputs

    def _encode(self, conversations: list[list[dict[str, Any]]], add_generation_prompt: bool) -> dict[str, torch.Tensor]:
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "return_dict": True,
        }
        processor_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
        }
        try:
            return self.processor.apply_chat_template(
                conversations,
                **template_kwargs,
                processor_kwargs=processor_kwargs,
            )
        except TypeError:
            legacy_kwargs = {**template_kwargs, **processor_kwargs}
            try:
                return self.processor.apply_chat_template(conversations, **legacy_kwargs)
            except TypeError:
                legacy_kwargs.pop("truncation", None)
                legacy_kwargs.pop("max_length", None)
                return self.processor.apply_chat_template(conversations, **legacy_kwargs)


def import_qwen3_vl_model_class():
    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen3VLForConditionalGeneration is unavailable. Install a recent Transformers build, "
            "for example: pip install -U git+https://github.com/huggingface/transformers"
        ) from exc


def dtype_from_args(args: argparse.Namespace) -> torch.dtype | str:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return "auto"


def from_pretrained_with_dtype(model_cls: Any, model_path: str, kwargs: dict[str, Any]):
    try:
        return model_cls.from_pretrained(model_path, **kwargs)
    except TypeError as exc:
        if "dtype" not in kwargs:
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
        return model_cls.from_pretrained(model_path, **fallback_kwargs)


def load_model(args: argparse.Namespace):
    model_cls = import_qwen3_vl_model_class()
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dtype": dtype_from_args(args),
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.load_in_4bit:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model_kwargs["device_map"] = {"": local_rank}
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    model = from_pretrained_with_dtype(model_cls, args.model_name_or_path, model_kwargs)
    model.config.use_cache = False

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    target_modules = [part.strip() for part in args.lora_target_modules.split(",") if part.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    if args.freeze_vision_lora:
        frozen = 0
        for name, param in model.named_parameters():
            name_l = name.lower()
            if "lora_" in name_l and any(marker in name_l for marker in VISION_NAME_MARKERS):
                param.requires_grad = False
                frozen += param.numel()
        if frozen:
            print(f"Froze {frozen:,} vision-side LoRA parameters")

    model.print_trainable_parameters()
    return model


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "optim": args.optim,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    if "save_safetensors" in params:
        kwargs["save_safetensors"] = True
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    if "gradient_checkpointing_kwargs" in params:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "ddp_find_unused_parameters" in params:
        kwargs["ddp_find_unused_parameters"] = False
    kwargs = {key: value for key, value in kwargs.items() if key in params}
    return TrainingArguments(**kwargs)


def build_trainer(
    model: torch.nn.Module,
    processor: Any,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    data_collator: VLDataCollator,
) -> Trainer:
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        kwargs["processing_class"] = processor
    else:
        kwargs["tokenizer"] = getattr(processor, "tokenizer", processor)
    return Trainer(**kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_dataset = JsonlVLSftDataset(args.train_jsonl, args.data_root, args.max_train_samples)
    eval_dataset = JsonlVLSftDataset(args.eval_jsonl, args.data_root, args.max_eval_samples)
    data_collator = VLDataCollator(processor=processor, max_length=args.max_length)

    if args.dry_run_batch:
        batch = data_collator([train_dataset[0], train_dataset[min(1, len(train_dataset) - 1)]])
        print({key: tuple(value.shape) for key, value in batch.items() if torch.is_tensor(value)})
        print("dry run ok")
        return

    model = load_model(args)
    training_args = build_training_args(args)
    trainer = build_trainer(
        model=model,
        processor=processor,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
