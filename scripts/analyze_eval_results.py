#!/usr/bin/env python3
"""Analyze dev evaluation results by difficulty, template, and function family."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Callable


try:
    from generate_sft_data import compute_visual_features
except Exception:
    compute_visual_features = None


R2_THRESHOLDS = (0.99, 0.95, 0.90, 0.80)
FUNC_RE = re.compile(r"np\.([A-Za-z_][A-Za-z0-9_]*)")


def threshold_key(threshold: float) -> str:
    return f"acc@{threshold:.2f}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expr_funcs(expr: str | None) -> str:
    funcs = sorted(set(FUNC_RE.findall(expr or "")))
    return ",".join(funcs) if funcs else "none"


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "null"
    return f"{value:.{digits}f}"


def pct(num: int, den: int) -> str:
    return f"{(num / den * 100 if den else 0.0):.1f}%"


def short(text: str | None, limit: int = 76) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def r2_bucket(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "null"
    if value >= 0.99:
        return ">=0.99"
    if value >= 0.95:
        return "0.95-0.99"
    if value >= 0.90:
        return "0.90-0.95"
    if value >= 0.80:
        return "0.80-0.90"
    if value >= 0.0:
        return "0.00-0.80"
    return "<0.00"


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def get_visual_features(sample_id: str, sample: dict[str, Any], true_expr: str | None, cache: dict[str, dict]) -> dict:
    if sample_id in cache:
        return cache[sample_id]
    features: dict[str, Any] = {"status": "unavailable"}
    if compute_visual_features is None:
        cache[sample_id] = features
        return features
    x_range = sample.get("image_x_range")
    if true_expr and isinstance(x_range, list) and len(x_range) == 2:
        try:
            features = compute_visual_features(true_expr, (float(x_range[0]), float(x_range[1])))
        except Exception as exc:
            features = {"status": "error", "error": str(exc)}
    cache[sample_id] = features
    return features


def enrich_results(results: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_by_id = {row["id"]: row for row in samples}
    visual_cache: dict[str, dict] = {}
    enriched: list[dict[str, Any]] = []
    for result in results:
        sample_id = str(result.get("id", ""))
        sample = sample_by_id.get(sample_id, {})
        config = sample.get("generation_config") or {}
        true_expr = result.get("true_expr") or sample.get("expression_numpy")
        pred_expr = result.get("predicted_expr")
        visual_features = get_visual_features(sample_id, sample, true_expr, visual_cache)
        symmetry = visual_features.get("symmetry") or {}
        row = {
            **result,
            "difficulty_name": sample.get("difficulty_name", "unknown"),
            "difficulty": sample.get("difficulty", "unknown"),
            "template_id": config.get("template_id", "unknown"),
            "true_funcs": expr_funcs(true_expr),
            "pred_funcs": expr_funcs(pred_expr),
            "hint_funcs": ",".join(sorted(h.replace("np.", "") for h in sample.get("function_hints", [])))
            or "none",
            "n_text_points": sample.get("n_text_points", "unknown"),
            "n_distractors": len(config.get("distractor_funcs") or []),
            "image_x_range": sample.get("image_x_range"),
            "true_expr": true_expr,
            "predicted_expr": pred_expr,
            "r2_bucket": r2_bucket(result.get("r2")),
            "visual_status": visual_features.get("status", "unknown"),
            "visual_symmetry": symmetry.get("type", "unknown"),
            "visual_zero_crossings": visual_features.get("zero_crossings", "unknown"),
            "visual_local_extrema": visual_features.get("local_extrema", "unknown"),
            "visual_monotonicity": visual_features.get("monotonicity", "unknown"),
            "visual_oscillatory": str(visual_features.get("oscillatory", "unknown")),
            "visual_estimated_period": visual_features.get("estimated_period"),
            "visual_changing_frequency": str(visual_features.get("changing_frequency", "unknown")),
            "visual_amplitude_trend": visual_features.get("amplitude_trend", "unknown"),
        }
        enriched.append(row)
    return enriched


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r2s = [row.get("r2") for row in rows]
    valid = [float(v) for v in r2s if isinstance(v, (int, float))]
    total = len(rows)
    summary: dict[str, Any] = {
        "n": total,
        "null": total - len(valid),
        "mean_r2": mean(valid),
        "median_r2": median(valid),
    }
    for threshold in R2_THRESHOLDS:
        summary[threshold_key(threshold)] = sum(1 for v in valid if v >= threshold)
    return summary


def print_table(
    title: str,
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], str],
    *,
    min_count: int = 1,
    limit: int | None = None,
) -> None:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)

    table = []
    for key, group_rows in groups.items():
        if len(group_rows) < min_count:
            continue
        s = summarize(group_rows)
        table.append((key, s))
    table.sort(key=lambda item: (item[1][threshold_key(0.99)] / max(item[1]["n"], 1), item[1]["n"], item[0]))
    if limit is not None:
        table = table[:limit]

    print(f"\n## {title}")
    print("group\tn\tacc@0.99\tacc@0.95\tacc@0.90\tacc@0.80\tnull\tmean_r2\tmedian_r2")
    for key, s in table:
        n = s["n"]
        print(
            "\t".join(
                [
                    key,
                    str(n),
                    pct(s[threshold_key(0.99)], n),
                    pct(s[threshold_key(0.95)], n),
                    pct(s[threshold_key(0.90)], n),
                    pct(s[threshold_key(0.80)], n),
                    pct(s["null"], n),
                    fmt_float(s["mean_r2"]),
                    fmt_float(s["median_r2"]),
                ]
            )
        )


def export_worst(rows: list[dict[str, Any]], path: Path, limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "r2",
        "mse",
        "difficulty_name",
        "template_id",
        "true_funcs",
        "pred_funcs",
        "hint_funcs",
        "n_text_points",
        "n_distractors",
        "r2_bucket",
        "visual_symmetry",
        "visual_oscillatory",
        "visual_changing_frequency",
        "visual_amplitude_trend",
        "visual_zero_crossings",
        "visual_local_extrema",
        "true_expr",
        "predicted_expr",
    ]
    worst = sorted(rows, key=lambda row: row.get("r2") if isinstance(row.get("r2"), (int, float)) else -999.0)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in worst[:limit]:
            writer.writerow({key: row.get(key) for key in fieldnames})


def print_worst(rows: list[dict[str, Any]], limit: int) -> None:
    worst = sorted(rows, key=lambda row: row.get("r2") if isinstance(row.get("r2"), (int, float)) else -999.0)
    print(f"\n## Worst {limit} Samples")
    print("id\tr2\tdifficulty\ttemplate\ttrue_funcs\tpred_funcs\ttrue_expr\tpredicted_expr")
    for row in worst[:limit]:
        print(
            "\t".join(
                [
                    str(row.get("id", "")),
                    fmt_float(row.get("r2")),
                    str(row.get("difficulty_name", "")),
                    str(row.get("template_id", "")),
                    str(row.get("true_funcs", "")),
                    str(row.get("pred_funcs", "")),
                    short(row.get("true_expr")),
                    short(row.get("predicted_expr")),
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        help="Path to eval_results_dev.jsonl produced by eval.py.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("data/task/dev/samples.jsonl"),
        help="Path to the dev samples.jsonl.",
    )
    parser.add_argument("--worst-limit", type=int, default=30)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV path for the worst samples.",
    )
    args = parser.parse_args()

    results = load_jsonl(args.results)
    samples = load_jsonl(args.samples)
    rows = enrich_results(results, samples)
    overall = summarize(rows)

    print("# Eval Analysis")
    print(f"results: {args.results}")
    print(f"samples: {args.samples}")
    print(
        "overall: "
        f"n={overall['n']} "
        f"acc@0.99={pct(overall[threshold_key(0.99)], overall['n'])} "
        f"acc@0.95={pct(overall[threshold_key(0.95)], overall['n'])} "
        f"acc@0.90={pct(overall[threshold_key(0.90)], overall['n'])} "
        f"acc@0.80={pct(overall[threshold_key(0.80)], overall['n'])} "
        f"null={pct(overall['null'], overall['n'])} "
        f"mean_r2={fmt_float(overall['mean_r2'])} "
        f"median_r2={fmt_float(overall['median_r2'])}"
    )

    print_table("By Difficulty", rows, lambda row: str(row["difficulty_name"]))
    print_table("By Template", rows, lambda row: str(row["template_id"]), min_count=3)
    print_table("By True Function Set", rows, lambda row: str(row["true_funcs"]), min_count=3)
    print_table("By Hint Function Set", rows, lambda row: str(row["hint_funcs"]), min_count=3)
    print_table("By Number Of Distractors", rows, lambda row: str(row["n_distractors"]))
    print_table("By R2 Bucket", rows, lambda row: str(row["r2_bucket"]))
    print_table("By Visual Symmetry", rows, lambda row: str(row["visual_symmetry"]))
    print_table("By Visual Oscillation", rows, lambda row: str(row["visual_oscillatory"]))
    print_table("By Visual Changing Frequency", rows, lambda row: str(row["visual_changing_frequency"]))
    print_table("By Visual Amplitude Trend", rows, lambda row: str(row["visual_amplitude_trend"]))
    print_table("By Visual Monotonicity", rows, lambda row: str(row["visual_monotonicity"]))
    print_worst(rows, args.worst_limit)

    if args.csv_out is not None:
        export_worst(rows, args.csv_out, args.worst_limit)
        print(f"\nWrote worst-sample CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
