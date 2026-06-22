#!/usr/bin/env python3
"""Build candidate-expression data for verifier-guided RL smoke tests.

The generated data is intentionally model-agnostic. On CPU it feeds the small
candidate policy in train_rl_candidate_policy.py. On GPU the same JSONL schema
can be populated with expressions sampled from Qwen, then used for DPO/GRPO
preparation.
"""

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

from rl_expr_utils import (  # noqa: E402
    build_candidate_features,
    compute_metrics,
    generic_distractors,
    mutate_expression_numbers,
    shaped_reward,
    summarize_jsonable_metrics,
    validate_expression,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_candidates(
    sample: dict[str, Any],
    rng: random.Random,
    max_candidates: int,
    baseline_expr: str | None,
) -> list[str]:
    true_expr = sample["expression_numpy"]
    candidates = [true_expr]
    if baseline_expr:
        candidates.append(baseline_expr)
    candidates.extend(mutate_expression_numbers(true_expr, max_mutations=max_candidates * 2))
    if baseline_expr and baseline_expr != true_expr:
        candidates.extend(mutate_expression_numbers(baseline_expr, max_mutations=max_candidates))
    candidates.extend(generic_distractors(sample.get("function_hints", [])))

    seen: set[str] = set()
    clean: list[str] = []
    for expr in candidates:
        if expr in seen or not validate_expression(expr):
            continue
        seen.add(expr)
        clean.append(expr)

    mandatory = [expr for expr in clean if expr == true_expr]
    if baseline_expr and baseline_expr != true_expr:
        mandatory.extend(expr for expr in clean if expr == baseline_expr)
    mandatory = list(dict.fromkeys(mandatory))
    other_candidates = [expr for expr in clean if expr not in set(mandatory)]
    rng.shuffle(other_candidates)
    selected = mandatory + other_candidates[: max(0, max_candidates - len(mandatory))]
    rng.shuffle(selected)
    return selected


def convert_sample(
    sample: dict[str, Any],
    rng: random.Random,
    max_candidates: int,
    baseline_expr: str | None,
) -> dict[str, Any]:
    candidates = []
    for expr in sample_candidates(sample, rng, max_candidates, baseline_expr):
        ref_metrics = compute_metrics(expr, sample.get("data_points_text", []))
        test_metrics = compute_metrics(expr, sample.get("test_points", []), max_points=50)
        candidates.append(
            {
                "expression": expr,
                "is_ground_truth": expr == sample["expression_numpy"],
                "is_baseline": baseline_expr is not None and expr == baseline_expr,
                "features": build_candidate_features(
                    expr,
                    sample.get("data_points_text", []),
                    sample.get("function_hints", []),
                ),
                "reference_metrics": summarize_jsonable_metrics(ref_metrics),
                "test_metrics": summarize_jsonable_metrics(test_metrics),
                "reward": shaped_reward(test_metrics),
            }
        )

    best_reward = max(candidate["reward"] for candidate in candidates)
    return {
        "id": sample["id"],
        "difficulty_name": sample.get("difficulty_name"),
        "template_id": (sample.get("generation_config") or {}).get("template_id"),
        "image_path": sample.get("image_path"),
        "function_hints": sample.get("function_hints", []),
        "reference_points": sample.get("data_points_text", []),
        "true_expr": sample["expression_numpy"],
        "candidates": candidates,
        "best_reward": best_reward,
        "has_positive": any(candidate["test_metrics"]["r2"] is not None and candidate["test_metrics"]["r2"] >= 0.99 for candidate in candidates),
    }


def load_baseline_predictions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    predictions: dict[str, str] = {}
    for row in rows:
        expr = row.get("predicted_expr")
        if row.get("id") and isinstance(expr, str) and expr.strip():
            predictions[str(row["id"])] = expr.strip()
    return predictions


def split_rows(rows: list[dict[str, Any]], rng: random.Random, eval_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    n_eval = max(1, int(round(len(rows) * eval_ratio))) if len(rows) > 1 else 0
    eval_indices = set(indices[:n_eval])
    train_rows = [row for idx, row in enumerate(rows) if idx not in eval_indices]
    eval_rows = [row for idx, row in enumerate(rows) if idx in eval_indices]
    return train_rows, eval_rows


def baseline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = []
    pass_count = 0
    covered = 0
    for row in rows:
        baseline_candidates = [candidate for candidate in row["candidates"] if candidate.get("is_baseline")]
        if not baseline_candidates:
            continue
        covered += 1
        candidate = baseline_candidates[0]
        rewards.append(float(candidate["reward"]))
        r2 = candidate["test_metrics"]["r2"]
        if r2 is not None and r2 >= 0.99:
            pass_count += 1
    return {
        "covered": covered,
        "mean_reward": sum(rewards) / covered if covered else None,
        "acc@0.99": pass_count / covered if covered else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path("data/task/dev/samples.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rl_cpu_smoke/data"))
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=None,
        help="Optional eval_results JSONL whose predicted_expr becomes the no-regression baseline candidate.",
    )
    args = parser.parse_args()

    if args.max_samples < 2:
        raise SystemExit("--max-samples must be at least 2")
    if args.max_candidates < 2:
        raise SystemExit("--max-candidates must be at least 2")
    if not 0.0 < args.eval_ratio < 1.0:
        raise SystemExit("--eval-ratio must be in (0, 1)")

    rng = random.Random(args.seed)
    samples = load_jsonl(args.samples)
    baseline_predictions = load_baseline_predictions(args.baseline_results)
    rng.shuffle(samples)
    selected = samples[: args.max_samples]
    rows = [
        convert_sample(
            sample,
            rng,
            args.max_candidates,
            baseline_predictions.get(sample["id"]),
        )
        for sample in selected
    ]
    train_rows, eval_rows = split_rows(rows, rng, args.eval_ratio)

    write_jsonl(args.out_dir / "rl_candidates.jsonl", rows)
    write_jsonl(args.out_dir / "rl_train.jsonl", train_rows)
    write_jsonl(args.out_dir / "rl_eval.jsonl", eval_rows)

    manifest = {
        "source_samples": str(args.samples),
        "num_total": len(rows),
        "num_train": len(train_rows),
        "num_eval": len(eval_rows),
        "max_candidates": args.max_candidates,
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "baseline_results": str(args.baseline_results) if args.baseline_results else None,
        "baseline_total": baseline_summary(rows),
        "baseline_train": baseline_summary(train_rows),
        "baseline_eval": baseline_summary(eval_rows),
        "schema": {
            "features": [
                "bias",
                "-log1p(reference_mse)",
                "-log1p(reference_max_abs_error)",
                "clipped_reference_r2",
                "function_hint_overlap",
                "-expression_length",
            ]
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote RL candidate data to {args.out_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
