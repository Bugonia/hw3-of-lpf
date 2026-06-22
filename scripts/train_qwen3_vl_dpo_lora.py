#!/usr/bin/env python3
"""Reference-free DPO-style LoRA training for Qwen3-VL preference data."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, get_cosine_schedule_with_warmup, set_seed

from peft import LoraConfig, PeftModel, get_peft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default="/inspire/hdd/project/generative-large-model/public/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter-name-or-path", default="/inspire/hdd/project/generative-large-model/public/hw3-of-lpf-best/qwen3_vl_v4_targeted_20260621/adapter")
    parser.add_argument("--train-jsonl", default="outputs/rl_dpo/data/preferences.jsonl")
    parser.add_argument("--output-dir", default="outputs/rl_dpo/qwen3_vl_dpo_lora")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["flash_attention_2", "sdpa", "eager", "auto"])
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--dry-run-batch", action="store_true", help="Build one collated DPO batch and exit before model loading.")
    return parser.parse_args()


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


class PreferenceDataset(Dataset):
    def __init__(self, path: str | Path, limit: int | None = None):
        self.rows = load_rows(Path(path), limit)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        prompt = copy.deepcopy(row["prompt_messages"])
        return {
            "id": row.get("id", str(idx)),
            "chosen_messages": prompt + [{"role": "assistant", "content": row["chosen"]}],
            "rejected_messages": copy.deepcopy(prompt) + [{"role": "assistant", "content": row["rejected"]}],
            "prompt_messages": prompt,
        }


@dataclass
class DpoDataCollator:
    processor: Any
    max_length: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        chosen = self._encode([feature["chosen_messages"] for feature in features], add_generation_prompt=False)
        rejected = self._encode([feature["rejected_messages"] for feature in features], add_generation_prompt=False)
        prompt = self._encode([feature["prompt_messages"] for feature in features], add_generation_prompt=True)
        prompt_lengths = prompt["attention_mask"].sum(dim=1).tolist()

        chosen["labels"] = self._labels(chosen, prompt_lengths)
        rejected["labels"] = self._labels(rejected, prompt_lengths)
        return {
            **{f"chosen_{key}": value for key, value in chosen.items()},
            **{f"rejected_{key}": value for key, value in rejected.items()},
        }

    def _labels(self, encoded: dict[str, torch.Tensor], prompt_lengths: list[int]) -> torch.Tensor:
        labels = encoded["input_ids"].clone()
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, : min(int(prompt_len), labels.shape[1])] = -100
        labels[encoded["attention_mask"] == 0] = -100
        return labels

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


def import_model_class():
    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError("Qwen3VLForConditionalGeneration unavailable in this Transformers build.") from exc


def dtype_from_args(args: argparse.Namespace) -> torch.dtype | str:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return "auto"


def from_pretrained_with_dtype(model_cls: Any, model_path: str, kwargs: dict[str, Any]):
    try:
        return model_cls.from_pretrained(model_path, **kwargs)
    except TypeError:
        fallback = dict(kwargs)
        fallback["torch_dtype"] = fallback.pop("dtype")
        return model_cls.from_pretrained(model_path, **fallback)


def load_model(args: argparse.Namespace):
    model_cls = import_model_class()
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dtype": dtype_from_args(args),
    }
    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    model = from_pretrained_with_dtype(model_cls, args.model_name_or_path, kwargs)
    model.config.use_cache = False

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.adapter_name_or_path and Path(args.adapter_name_or_path).exists():
        model = PeftModel.from_pretrained(model, args.adapter_name_or_path, is_trainable=True)
        print(f"Loaded trainable adapter: {args.adapter_name_or_path}")
    else:
        target_modules = [part.strip() for part in args.lora_target_modules.split(",") if part.strip()]
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def sequence_logps(model: torch.nn.Module, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    inputs = {
        key[len(prefix) + 1 :]: value
        for key, value in batch.items()
        if key.startswith(prefix + "_") and key != f"{prefix}_labels"
    }
    labels = batch[f"{prefix}_labels"]
    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:].clone()
    loss_mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    token_logps = torch.gather(F.log_softmax(logits, dim=-1), dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp_min(1)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def save_adapter(model: torch.nn.Module, processor: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dataset = PreferenceDataset(args.train_jsonl, args.max_samples)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    collator = DpoDataCollator(processor=processor, max_length=args.max_length)

    if args.dry_run_batch:
        batch = collator([dataset[0]])
        print({key: tuple(value.shape) for key, value in batch.items() if torch.is_tensor(value)})
        print("dry run ok")
        return

    model = load_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    steps_per_epoch = max(1, len(loader) // max(args.gradient_accumulation_steps, 1))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.num_train_epochs):
        for batch_idx, batch in enumerate(loader):
            batch = move_batch(batch, device)
            chosen_logps = sequence_logps(model, batch, "chosen")
            rejected_logps = sequence_logps(model, batch, "rejected")
            logits = args.beta * (chosen_logps - rejected_logps)
            loss = -F.logsigmoid(logits).mean()
            (loss / args.gradient_accumulation_steps).backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.logging_steps == 0:
                    margin = float((chosen_logps - rejected_logps).mean().detach().cpu())
                    print(json.dumps({"step": global_step, "loss": float(loss.detach().cpu()), "logp_margin": margin}))
                if global_step % args.save_steps == 0:
                    save_adapter(model, processor, Path(args.output_dir) / f"checkpoint-{global_step}")
                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break

    save_adapter(model, processor, Path(args.output_dir))
    summary = {
        "global_step": global_step,
        "train_rows": len(dataset),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "dpo_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
