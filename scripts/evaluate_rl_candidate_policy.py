#!/usr/bin/env python3
"""Evaluate a trained candidate-expression RL policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from train_rl_candidate_policy import CandidatePolicy, evaluate, load_jsonl, row_tensors


@torch.no_grad()
def write_predictions(policy: CandidatePolicy, rows: list[dict[str, Any]], device: torch.device, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            features, _rewards = row_tensors(row, device)
            logits = policy(features)
            selected_idx = int(torch.argmax(logits).item())
            selected = row["candidates"][selected_idx]
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "template_id": row.get("template_id"),
                        "difficulty_name": row.get("difficulty_name"),
                        "true_expr": row["true_expr"],
                        "predicted_expr": selected["expression"],
                        "selected_idx": selected_idx,
                        "reward": selected["reward"],
                        "r2": selected["test_metrics"]["r2"],
                        "mse": selected["test_metrics"]["mse"],
                        "is_ground_truth": selected["is_ground_truth"],
                        "is_baseline": selected.get("is_baseline", False),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/rl_cpu_smoke/policy/candidate_policy.pt"))
    parser.add_argument("--eval-jsonl", type=Path, default=Path("outputs/rl_cpu_smoke/data/rl_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rl_cpu_smoke/eval"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    except Exception:
        # Local smoke-test checkpoints are produced by this repo. Fall back for
        # PyTorch versions whose default weights_only=True rejects metadata.
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = CandidatePolicy(int(checkpoint["feature_dim"])).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    rows = load_jsonl(args.eval_jsonl)
    summary = evaluate(policy, rows, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_predictions(policy, rows, device, args.output_dir / "eval_predictions.jsonl")

    print(json.dumps(summary, indent=2))
    print(f"Wrote evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
