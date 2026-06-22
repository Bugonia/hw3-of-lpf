#!/usr/bin/env python3
"""Train a tiny verifier-guided candidate policy with REINFORCE.

This is the CPU smoke-test trainer for the RL pipeline. It does not replace
Qwen fine-tuning; it verifies that candidate data, verifier rewards, policy
optimization, checkpointing, and evaluation are wired correctly.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_tensors(row: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor([candidate["features"] for candidate in row["candidates"]], dtype=torch.float32, device=device)
    rewards = torch.tensor([candidate["reward"] for candidate in row["candidates"]], dtype=torch.float32, device=device)
    return features, rewards


class CandidatePolicy(torch.nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.scorer = torch.nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scorer(features).squeeze(-1)


@torch.no_grad()
def evaluate(policy: CandidatePolicy, rows: list[dict[str, Any]], device: torch.device) -> dict[str, float]:
    greedy_rewards = []
    oracle_rewards = []
    pass_at_099 = 0
    selected_gt = 0
    for row in rows:
        features, rewards = row_tensors(row, device)
        logits = policy(features)
        selected_idx = int(torch.argmax(logits).item())
        selected = row["candidates"][selected_idx]
        reward = float(rewards[selected_idx].item())
        greedy_rewards.append(reward)
        oracle_rewards.append(float(torch.max(rewards).item()))
        r2 = selected["test_metrics"]["r2"]
        if r2 is not None and r2 >= 0.99:
            pass_at_099 += 1
        if selected.get("is_ground_truth"):
            selected_gt += 1

    n = max(len(rows), 1)
    return {
        "mean_reward": sum(greedy_rewards) / n,
        "oracle_mean_reward": sum(oracle_rewards) / n,
        "acc@0.99": pass_at_099 / n,
        "ground_truth_selection_rate": selected_gt / n,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_rows = load_jsonl(args.train_jsonl)
    eval_rows = load_jsonl(args.eval_jsonl)
    if not train_rows:
        raise ValueError(f"No rows in {args.train_jsonl}")
    feature_dim = len(train_rows[0]["candidates"][0]["features"])

    policy = CandidatePolicy(feature_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    before_train = evaluate(policy, train_rows, device)
    before_eval = evaluate(policy, eval_rows, device)
    history = []

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        total_loss = 0.0
        total_reward = 0.0
        for row in train_rows:
            features, rewards = row_tensors(row, device)
            logits = policy(features)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            reward = rewards[action]
            baseline = torch.sum(torch.softmax(logits.detach(), dim=0) * rewards)
            advantage = reward - baseline
            entropy_bonus = args.entropy_coef * dist.entropy()
            loss = -(dist.log_prob(action) * advantage.detach()) - entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item())
            total_reward += float(reward.item())

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            eval_metrics = evaluate(policy, eval_rows, device)
            item = {
                "epoch": epoch,
                "train_sampled_reward": total_reward / max(len(train_rows), 1),
                "train_loss": total_loss / max(len(train_rows), 1),
                "eval": eval_metrics,
            }
            history.append(item)
            print(json.dumps(item, indent=2))

    after_train = evaluate(policy, train_rows, device)
    after_eval = evaluate(policy, eval_rows, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    checkpoint = {
        "model_state_dict": policy.state_dict(),
        "feature_dim": feature_dim,
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "args": serialized_args,
    }
    torch.save(checkpoint, args.output_dir / "candidate_policy.pt")

    summary = {
        "before_train": before_train,
        "before_eval": before_eval,
        "after_train": after_train,
        "after_eval": after_eval,
        "history": history,
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=Path("outputs/rl_cpu_smoke/data/rl_train.jsonl"))
    parser.add_argument("--eval-jsonl", type=Path, default=Path("outputs/rl_cpu_smoke/data/rl_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rl_cpu_smoke/policy"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = train(args)
    print("Training summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
