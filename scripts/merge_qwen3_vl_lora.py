#!/usr/bin/env python3
"""Merge a Qwen3-VL LoRA adapter into a full model directory for eval.py/vLLM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Qwen3-VL LoRA adapter")
    parser.add_argument(
        "--base-model",
        default="/inspire/hdd/project/generative-large-model/public/models/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--adapter",
        default="/inspire/hdd/project/generative-large-model/public/outputs/qwen3_vl_stage1_lora",
    )
    parser.add_argument(
        "--output-dir",
        default="/inspire/hdd/project/generative-large-model/public/outputs/qwen3_vl_stage1_merged",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--attn-implementation", default="auto", choices=["flash_attention_2", "sdpa", "eager", "auto"])
    return parser.parse_args()


def import_qwen3_vl_model_class():
    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen3VLForConditionalGeneration is unavailable. Install a recent Transformers build, "
            "for example: pip install -U git+https://github.com/huggingface/transformers"
        ) from exc


def dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cls = import_qwen3_vl_model_class()
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": dtype(args.dtype),
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    try:
        base = model_cls.from_pretrained(args.base_model, **model_kwargs)
    except TypeError:
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        base = model_cls.from_pretrained(args.base_model, **model_kwargs)

    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")


if __name__ == "__main__":
    main()
