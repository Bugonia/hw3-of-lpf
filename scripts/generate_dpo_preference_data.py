#!/usr/bin/env python3
"""Generate verifier-guided chosen/rejected data for Qwen3-VL DPO training."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_sft_data import DEFAULT_PROMPT  # noqa: E402
from rl_expr_utils import compute_metrics, generic_distractors, mutate_expression_numbers, shaped_reward, validate_expression  # noqa: E402


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(function_hints: list[str], data_points: list[list[float]]) -> str:
    hints_text = "Available functions: " + ", ".join(function_hints) + "\n" if function_hints else ""
    points_text = "Reference points: " + "  ".join(
        f"({x:.4f}, {y:.4f})" for x, y in data_points
    ) + "\n"
    return DEFAULT_PROMPT.format(function_hints=hints_text, data_points=points_text, axis_note="")


def tool_answer(expr: str, *, short_reasoning: bool) -> str:
    call = "<tool_call>\n" + json.dumps(
        {"name": "submit_expression", "arguments": {"expression": expr}},
        ensure_ascii=False,
    ) + "\n</tool_call>"
    if not short_reasoning:
        return call
    return (
        "<think>\n"
        "I use the plotted structure for the family and verify the expression against the reference points before submitting.\n"
        "</think>\n"
        + call
    )


def candidate_rejections(sample: dict[str, Any], baseline_expr: str | None) -> list[dict[str, Any]]:
    true_expr = sample["expression_numpy"]
    raw_candidates = []
    if baseline_expr and baseline_expr != true_expr:
        raw_candidates.append(("baseline", baseline_expr))
        raw_candidates.extend(("baseline_mutation", expr) for expr in mutate_expression_numbers(baseline_expr, 8))
    raw_candidates.extend(("truth_mutation", expr) for expr in mutate_expression_numbers(true_expr, 16))
    raw_candidates.extend(("generic", expr) for expr in generic_distractors(sample.get("function_hints", [])))

    seen = {true_expr}
    candidates = []
    for source, expr in raw_candidates:
        if expr in seen or not validate_expression(expr):
            continue
        seen.add(expr)
        test_metrics = compute_metrics(expr, sample.get("test_points", []), max_points=50)
        ref_metrics = compute_metrics(expr, sample.get("data_points_text", []))
        reward = shaped_reward(test_metrics)
        candidates.append(
            {
                "source": source,
                "expression": expr,
                "reward": reward,
                "r2": test_metrics.r2,
                "reference_r2": ref_metrics.r2,
                "valid": test_metrics.valid,
            }
        )
    candidates.sort(key=lambda item: (item["reward"], item["r2"] if item["r2"] is not None else -999.0))
    return candidates


def load_baseline_predictions(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    predictions = {}
    for row in load_jsonl(path):
        expr = row.get("predicted_expr")
        if row.get("id") and isinstance(expr, str) and expr.strip():
            predictions[str(row["id"])] = expr.strip()
    return predictions


def convert_sample(
    sample: dict[str, Any],
    data_root: Path,
    baseline_predictions: dict[str, str],
    short_reasoning: bool,
    rejection_mode: str,
) -> dict[str, Any] | None:
    true_expr = sample["expression_numpy"]
    true_metrics = compute_metrics(true_expr, sample.get("test_points", []), max_points=50)
    if not true_metrics.valid or true_metrics.r2 is None or true_metrics.r2 < 0.999:
        return None

    rejected_candidates = candidate_rejections(sample, baseline_predictions.get(sample["id"]))
    rejected_candidates = [item for item in rejected_candidates if item["reward"] < shaped_reward(true_metrics)]
    if not rejected_candidates:
        return None
    if rejection_mode == "worst":
        rejected = rejected_candidates[0]
    elif rejection_mode == "hardest":
        rejected = rejected_candidates[-1]
    else:
        raise ValueError(f"unknown rejection_mode: {rejection_mode}")

    image_path = sample.get("image_path", "")
    if image_path and not str(image_path).startswith(("/", "http://", "https://", "file://")):
        image_path = str((data_root / image_path).resolve())

    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": build_prompt(sample.get("function_hints", []), sample.get("data_points_text", [])),
                },
            ],
        }
    ]

    return {
        "id": sample["id"],
        "difficulty_name": sample.get("difficulty_name"),
        "template_id": (sample.get("generation_config") or {}).get("template_id"),
        "prompt_messages": prompt_messages,
        "chosen": tool_answer(true_expr, short_reasoning=short_reasoning),
        "rejected": tool_answer(rejected["expression"], short_reasoning=short_reasoning),
        "chosen_expr": true_expr,
        "rejected_expr": rejected["expression"],
        "chosen_reward": shaped_reward(true_metrics),
        "rejected_reward": rejected["reward"],
        "rejected_source": rejected["source"],
        "rejected_r2": rejected["r2"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/rl_dpo/data/preferences.jsonl"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--short-reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rejection-mode",
        choices=["hardest", "worst"],
        default="hardest",
        help="Use hardest valid negative by default. 'worst' is only for debugging and can over-train.",
    )
    args = parser.parse_args()

    data_root = args.data_root or args.samples.parent
    rng = random.Random(args.seed)
    samples = load_jsonl(args.samples)
    rng.shuffle(samples)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    baseline_predictions = load_baseline_predictions(args.baseline_results)
    rows = []
    for sample in samples:
        row = convert_sample(sample, data_root, baseline_predictions, args.short_reasoning, args.rejection_mode)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("No DPO rows generated; check samples and candidate construction.")

    write_jsonl(args.out, rows)
    manifest = {
        "source_samples": str(args.samples),
        "data_root": str(data_root),
        "baseline_results": str(args.baseline_results) if args.baseline_results else None,
        "num_rows": len(rows),
        "seed": args.seed,
        "short_reasoning": args.short_reasoning,
        "rejection_mode": args.rejection_mode,
        "output": str(args.out),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
