#!/usr/bin/env python3
"""Compare two eval.py result files sample-by-sample and by template."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any


THRESHOLD = 0.99


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def r2_value(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    value = row.get("r2")
    return float(value) if isinstance(value, (int, float)) else None


def pass_at(row: dict[str, Any] | None, threshold: float = THRESHOLD) -> bool:
    value = r2_value(row)
    return value is not None and value >= threshold


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "null"
    return f"{value:.{digits}f}"


def pct(num: int, den: int) -> str:
    return f"{(num / den * 100 if den else 0.0):.1f}%"


def sample_meta(samples_path: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(samples_path):
        config = row.get("generation_config") or {}
        meta[row["id"]] = {
            "difficulty": row.get("difficulty_name", "unknown"),
            "template": config.get("template_id", "unknown"),
            "true_expr": row.get("expression_numpy", ""),
        }
    return meta


def summarize_group(items: list[tuple[dict[str, Any], dict[str, Any]]], threshold: float) -> dict[str, Any]:
    old_pass = sum(1 for old, _new in items if pass_at(old, threshold))
    new_pass = sum(1 for _old, new in items if pass_at(new, threshold))
    deltas = []
    for old, new in items:
        old_r2 = r2_value(old)
        new_r2 = r2_value(new)
        if old_r2 is not None and new_r2 is not None:
            deltas.append(new_r2 - old_r2)
    return {
        "n": len(items),
        "old_pass": old_pass,
        "new_pass": new_pass,
        "delta_pass": new_pass - old_pass,
        "mean_delta_r2": sum(deltas) / len(deltas) if deltas else None,
        "median_delta_r2": statistics.median(deltas) if deltas else None,
        "fixed": sum(1 for old, new in items if not pass_at(old, threshold) and pass_at(new, threshold)),
        "regressed": sum(1 for old, new in items if pass_at(old, threshold) and not pass_at(new, threshold)),
    }


def short(text: str | None, limit: int = 92) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_results", type=Path)
    parser.add_argument("new_results", type=Path)
    parser.add_argument("--samples", type=Path, default=Path("data/task/dev/samples.jsonl"))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    old_by_id = {row["id"]: row for row in load_jsonl(args.old_results)}
    new_by_id = {row["id"]: row for row in load_jsonl(args.new_results)}
    meta_by_id = sample_meta(args.samples)

    ids = sorted(set(old_by_id) & set(new_by_id))
    pairs = [(old_by_id[sid], new_by_id[sid]) for sid in ids]
    overall = summarize_group(pairs, args.threshold)

    print("# Eval Comparison")
    print(f"old: {args.old_results}")
    print(f"new: {args.new_results}")
    print(f"threshold: {args.threshold}")
    print(
        "overall: "
        f"n={overall['n']} "
        f"old={pct(overall['old_pass'], overall['n'])} "
        f"new={pct(overall['new_pass'], overall['n'])} "
        f"delta={overall['delta_pass']:+d} "
        f"fixed={overall['fixed']} "
        f"regressed={overall['regressed']} "
        f"mean_delta_r2={fmt(overall['mean_delta_r2'])} "
        f"median_delta_r2={fmt(overall['median_delta_r2'])}"
    )

    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)
    for sid in ids:
        meta = meta_by_id.get(sid, {})
        groups[str(meta.get("template", "unknown"))].append((old_by_id[sid], new_by_id[sid]))

    table = [(template, summarize_group(items, args.threshold)) for template, items in groups.items()]
    table.sort(key=lambda item: (item[1]["delta_pass"], item[1]["new_pass"] / max(item[1]["n"], 1), item[0]))
    print("\n## By Template")
    print("template\tn\told\tnew\tdelta\tfixed\tregressed\tmean_delta_r2\tmedian_delta_r2")
    for template, summary in table:
        n = summary["n"]
        print(
            "\t".join(
                [
                    template,
                    str(n),
                    pct(summary["old_pass"], n),
                    pct(summary["new_pass"], n),
                    f"{summary['delta_pass']:+d}",
                    str(summary["fixed"]),
                    str(summary["regressed"]),
                    fmt(summary["mean_delta_r2"]),
                    fmt(summary["median_delta_r2"]),
                ]
            )
        )

    rows = []
    for sid in ids:
        old = old_by_id[sid]
        new = new_by_id[sid]
        old_r2 = r2_value(old)
        new_r2 = r2_value(new)
        if old_r2 is None or new_r2 is None:
            delta = None
        else:
            delta = new_r2 - old_r2
        meta = meta_by_id.get(sid, {})
        rows.append((sid, old, new, old_r2, new_r2, delta, meta))

    print(f"\n## Biggest Fixes")
    print("id\ttemplate\tdifficulty\told_r2\tnew_r2\tdelta\ttrue_expr\tnew_pred")
    fixes = [row for row in rows if not pass_at(row[1], args.threshold) and pass_at(row[2], args.threshold)]
    fixes.sort(key=lambda row: row[5] if row[5] is not None else -999.0, reverse=True)
    for sid, _old, new, old_r2, new_r2, delta, meta in fixes[: args.limit]:
        print(
            "\t".join(
                [
                    sid,
                    str(meta.get("template", "")),
                    str(meta.get("difficulty", "")),
                    fmt(old_r2),
                    fmt(new_r2),
                    fmt(delta),
                    short(meta.get("true_expr")),
                    short(new.get("predicted_expr")),
                ]
            )
        )

    print(f"\n## Regressions")
    print("id\ttemplate\tdifficulty\told_r2\tnew_r2\tdelta\ttrue_expr\tnew_pred")
    regressions = [row for row in rows if pass_at(row[1], args.threshold) and not pass_at(row[2], args.threshold)]
    regressions.sort(key=lambda row: row[5] if row[5] is not None else 0.0)
    for sid, _old, new, old_r2, new_r2, delta, meta in regressions[: args.limit]:
        print(
            "\t".join(
                [
                    sid,
                    str(meta.get("template", "")),
                    str(meta.get("difficulty", "")),
                    fmt(old_r2),
                    fmt(new_r2),
                    fmt(delta),
                    short(meta.get("true_expr")),
                    short(new.get("predicted_expr")),
                ]
            )
        )


if __name__ == "__main__":
    main()
