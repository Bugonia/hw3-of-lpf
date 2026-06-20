#!/usr/bin/env python3
"""Generate stage-1 symbolic-regression SFT data.

The generator mirrors the released dev distribution: each sample contains a
curve image, function hints with distractors, Chebyshev-sampled reference
points, the ground-truth numpy expression, and an assistant tool-call answer.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hw3_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PROMPT = """\
You are a symbolic regression expert. Given a function curve plot and reference information, infer the closed-form expression for f(x) and call submit_expression with a numpy expression.

{function_hints}{data_points}{axis_note}
Carefully analyze the curve's shape: periodicity, growth/decay, symmetry, asymptotes, and the location of peaks and zeros. Cross-check your hypothesis against the reference points before answering.

You may reason inside <think>...</think> before calling the tool.
Use only numpy (np.) functions; the variable must be named x."""

ALL_FUNCTION_HINTS = [
    "np.sin",
    "np.cos",
    "np.exp",
    "np.sqrt",
    "np.log",
    "np.abs",
    "np.tanh",
    "np.tan",
    "np.arctan",
    "np.arccos",
    "np.arcsin",
    "np.sinh",
    "np.cosh",
]

DIFFICULTY_TO_LEVEL = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
    "expert": 4,
    "extreme": 5,
}


@dataclass(frozen=True)
class Template:
    template_id: str
    difficulty_name: str
    param_choices: dict[str, list[float]]
    x_ranges: list[tuple[float, float]]
    true_hints: list[str]
    expression_builder: Callable[[dict[str, float]], str]

    @property
    def difficulty(self) -> int:
        return DIFFICULTY_TO_LEVEL[self.difficulty_name]


def fmt_num(value: float) -> str:
    """Format numbers the way the released expressions usually do."""
    if abs(value - int(value)) < 1e-12:
        return str(int(value))
    return f"{value:.6g}"


def build_templates() -> list[Template]:
    t = Template
    return [
        t(
            "L1_sin",
            "easy",
            {"a": [0.5, 1.0, 1.5, 2.0, 3.0], "b": [1.0, 2.0, 3.0]},
            [(-math.pi, math.pi), (-2 * math.pi, 2 * math.pi)],
            ["np.sin"],
            lambda p: f"{fmt_num(p['a'])} * np.sin({fmt_num(p['b'])} * x)",
        ),
        t(
            "L1_cos",
            "easy",
            {"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]},
            [(-math.pi, math.pi), (-2 * math.pi, 2 * math.pi)],
            ["np.cos"],
            lambda p: f"{fmt_num(p['a'])} * np.cos({fmt_num(p['b'])} * x)",
        ),
        t(
            "L1_exp_grow",
            "easy",
            {"a": [1.0, 2.0], "b": [0.5, 1.0, 1.5]},
            [(-2.0, 2.0), (-1.0, 3.0)],
            ["np.exp"],
            lambda p: f"{fmt_num(p['a'])} * np.exp({fmt_num(p['b'])} * x)",
        ),
        t(
            "L1_exp_decay",
            "easy",
            {"a": [1.0, 2.0, 3.0], "b": [-2.0, -1.5, -1.0, -0.5]},
            [(-1.0, 4.0), (0.0, 5.0)],
            ["np.exp"],
            lambda p: f"{fmt_num(p['a'])} * np.exp({fmt_num(p['b'])} * x)",
        ),
        t(
            "L1_sqrt",
            "easy",
            {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0]},
            [(0.0, 4.0), (0.0, 6.0)],
            ["np.sqrt"],
            lambda p: f"{fmt_num(p['a'])} * np.sqrt({fmt_num(p['b'])} * x)",
        ),
        t(
            "L1_poly",
            "easy",
            {"a": [0.5, 1.0, 2.0, 3.0], "n": [2.0, 3.0]},
            [(-2.0, 2.0), (-3.0, 3.0)],
            [],
            lambda p: f"{fmt_num(p['a'])} * x ** {fmt_num(p['n'])}",
        ),
        t(
            "L2_gaussian",
            "medium",
            {"a": [1.0, 2.0, 3.0], "b": [0.5, 1.0, 2.0], "c": [0.0, 0.5, 1.0]},
            [(-3.0, 3.0), (-4.0, 4.0)],
            ["np.exp"],
            lambda p: f"{fmt_num(p['a'])} * np.exp(-{fmt_num(p['b'])} * x ** 2) + {fmt_num(p['c'])}",
        ),
        t(
            "L2_sin_plus_linear",
            "medium",
            {
                "a": [1.0, 2.0, 3.0],
                "b": [2.0, 3.0],
                "c": [-0.5, 0.5, 1.0],
                "d": [-1.0, 0.0, 1.0],
            },
            [(-4.0, 4.0), (-6.0, 6.0)],
            ["np.sin"],
            lambda p: f"{fmt_num(p['a'])} * np.sin({fmt_num(p['b'])} * x) + {fmt_num(p['c'])} * x + {fmt_num(p['d'])}",
        ),
        t(
            "L2_sin_full",
            "medium",
            {
                "a": [1.0, 2.0, 3.0],
                "b": [1.0, 2.0, 3.0],
                "c": [-0.5, 0.5, 1.0, 1.5],
                "d": [-2.0, -1.0, 0.0, 1.0],
            },
            [(-math.pi, math.pi), (-2 * math.pi, 2 * math.pi)],
            ["np.sin"],
            lambda p: f"{fmt_num(p['a'])} * np.sin({fmt_num(p['b'])} * x + {fmt_num(p['c'])}) + {fmt_num(p['d'])}",
        ),
        t(
            "L2_sin_cos",
            "medium",
            {
                "a": [1.0, 2.0, 3.0],
                "b": [1.0, 2.0],
                "c": [1.0, 2.0, 3.0],
                "d": [1.0, 2.0, 3.0],
            },
            [(-math.pi, math.pi), (-2 * math.pi, 2 * math.pi)],
            ["np.sin", "np.cos"],
            lambda p: f"{fmt_num(p['a'])} * np.sin({fmt_num(p['b'])} * x) + {fmt_num(p['c'])} * np.cos({fmt_num(p['d'])} * x)",
        ),
        t(
            "L2_log",
            "medium",
            {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0], "c": [-2.0, -1.0, 0.0, 1.0]},
            [(-4.0, 4.0), (-5.0, 5.0)],
            ["np.log"],
            lambda p: f"{fmt_num(p['a'])} * np.log(np.abs({fmt_num(p['b'])} * x) + 1) + {fmt_num(p['c'])}",
        ),
        t(
            "L2_exp_offset",
            "medium",
            {"a": [1.0, 2.0], "b": [-1.0, -0.5, 0.5, 1.0], "c": [-1.0, 0.0, 1.0, 2.0]},
            [(-2.0, 3.0), (-1.0, 4.0)],
            ["np.exp"],
            lambda p: f"{fmt_num(p['a'])} * np.exp({fmt_num(p['b'])} * x) + {fmt_num(p['c'])}",
        ),
        t(
            "L2_cos_full",
            "medium",
            {
                "a": [1.0, 2.0, 3.0],
                "b": [2.0, 3.0],
                "c": [-1.0, -0.5, 0.5, 1.0],
                "d": [-1.0, 0.0, 1.0],
            },
            [(-math.pi, math.pi), (-2 * math.pi, 2 * math.pi)],
            ["np.cos"],
            lambda p: f"{fmt_num(p['a'])} * np.cos({fmt_num(p['b'])} * x + {fmt_num(p['c'])}) + {fmt_num(p['d'])}",
        ),
        t(
            "L3_chirp",
            "hard",
            {"a": [1.0, 2.0, 3.0], "b": [0.5, 1.0, 2.0], "c": [0.0, 0.5, 1.0]},
            [(-4.0, 4.0), (-5.0, 5.0)],
            ["np.sin"],
            lambda p: f"{fmt_num(p['a'])} * np.sin({fmt_num(p['b'])} * x ** 2 + {fmt_num(p['c'])})",
        ),
        t(
            "L3_sqrt_sin",
            "hard",
            {"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0], "c": [0.0, 0.5, 1.0]},
            [(-5.0, 5.0)],
            ["np.sqrt", "np.sin"],
            lambda p: f"{fmt_num(p['a'])} * np.sqrt(np.abs(x)) * np.sin({fmt_num(p['b'])} * x + {fmt_num(p['c'])})",
        ),
        t(
            "L3_damped_osc",
            "hard",
            {"a": [1.0, 2.0, 3.0], "b": [0.3, 0.5, 0.8, 1.0], "c": [2.0, 3.0, 4.0, 5.0]},
            [(0.0, 6.0), (0.0, 8.0)],
            ["np.exp", "np.cos"],
            lambda p: f"{fmt_num(p['a'])} * np.exp(-{fmt_num(p['b'])} * x) * np.cos({fmt_num(p['c'])} * x)",
        ),
        t(
            "L3_gauss_sin",
            "hard",
            {"a": [1.0, 2.0, 3.0], "b": [0.2, 0.5, 1.0], "c": [2.0, 3.0, 4.0]},
            [(-4.0, 4.0), (-5.0, 5.0)],
            ["np.exp", "np.sin"],
            lambda p: f"{fmt_num(p['a'])} * np.exp(-{fmt_num(p['b'])} * x ** 2) * np.sin({fmt_num(p['c'])} * x)",
        ),
        t(
            "L3_beat",
            "hard",
            {"a": [2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [1.0, 2.0]},
            [(-math.pi, math.pi), (-2 * math.pi, 2 * math.pi)],
            ["np.sin", "np.cos"],
            lambda p: f"{fmt_num(p['a'])} * np.sin({fmt_num(p['b'])} * x) * np.cos({fmt_num(p['c'])} * x)",
        ),
        t(
            "L3_growing_osc",
            "hard",
            {"a": [0.5, 1.0, 1.5], "b": [1.0, 2.0, 3.0]},
            [(-4.0, 4.0), (-5.0, 5.0)],
            ["np.sin"],
            lambda p: f"{fmt_num(p['a'])} * x * np.sin({fmt_num(p['b'])} * x)",
        ),
        t(
            "L4_log_sin",
            "expert",
            {"a": [1.0, 2.0], "b": [1.0, 2.0], "c": [0.5, 1.0, 2.0], "d": [1.0, 2.0, 3.0]},
            [(-6.0, 6.0)],
            ["np.log", "np.sin"],
            lambda p: f"{fmt_num(p['a'])} * np.log(np.abs({fmt_num(p['b'])} + {fmt_num(p['c'])} * np.sin({fmt_num(p['d'])} * x)) + 1)",
        ),
        t(
            "L4_three_terms",
            "expert",
            {
                "a": [1.0, 2.0, 3.0],
                "b": [1.0, 2.0, 3.0],
                "c": [1.0, 2.0],
                "d": [0.5, 1.0, 2.0],
                "e": [-1.0, -0.5, 0.5, 1.0],
            },
            [(-4.0, 4.0), (-5.0, 5.0)],
            ["np.sin", "np.exp"],
            lambda p: f"{fmt_num(p['a'])} * np.sin({fmt_num(p['b'])} * x) + {fmt_num(p['c'])} * np.exp(-{fmt_num(p['d'])} * x ** 2) + {fmt_num(p['e'])} * x",
        ),
        t(
            "L4_exp_chirp",
            "expert",
            {"a": [1.0, 2.0, 3.0], "b": [0.3, 0.5, 0.8], "c": [0.5, 1.0, 2.0]},
            [(0.0, 6.0), (0.0, 8.0)],
            ["np.exp", "np.sin"],
            lambda p: f"{fmt_num(p['a'])} * np.exp(-{fmt_num(p['b'])} * x) * np.sin({fmt_num(p['c'])} * x ** 2)",
        ),
        t(
            "L4_sqrt_cos_sq",
            "expert",
            {"a": [1.0, 2.0, 3.0], "b": [0.5, 1.0, 2.0], "c": [2.0, 3.0, 4.0]},
            [(-4.0, 4.0), (-5.0, 5.0)],
            ["np.sqrt", "np.cos"],
            lambda p: f"np.sqrt(np.abs({fmt_num(p['a'])} * np.cos({fmt_num(p['b'])} * x ** 2) + {fmt_num(p['c'])}))",
        ),
        t(
            "L4_sin_of_exp",
            "expert",
            {"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0], "c": [0.3, 0.5, 0.8]},
            [(-2.0, 3.0), (-1.0, 4.0)],
            ["np.sin", "np.exp"],
            lambda p: f"{fmt_num(p['a'])} * np.sin({fmt_num(p['b'])} * np.exp({fmt_num(p['c'])} * x))",
        ),
        t(
            "L5_sqrt_chirp_poly",
            "extreme",
            {"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0], "c": [1.0, 2.0, 3.0], "d": [-0.5, 0.5, 1.0]},
            [(-4.0, 4.0), (-5.0, 5.0)],
            ["np.sin", "np.sqrt"],
            lambda p: f"{fmt_num(p['a'])} * np.sin({fmt_num(p['b'])} * np.sqrt(np.abs({fmt_num(p['c'])} * x))) + {fmt_num(p['d'])} * x ** 2",
        ),
        t(
            "L5_tanh_nested",
            "extreme",
            {
                "a": [1.0, 2.0],
                "b": [1.0, 2.0, 3.0],
                "c": [1.0, 2.0, 3.0],
                "d": [-1.0, 0.0, 1.0],
                "e": [0.5, 1.0],
                "f": [2.0, 3.0, 4.0],
            },
            [(-4.0, 4.0), (-6.0, 6.0)],
            ["np.tanh", "np.sin"],
            lambda p: f"{fmt_num(p['a'])} * np.tanh({fmt_num(p['b'])} * np.sin({fmt_num(p['c'])} * x) + {fmt_num(p['d'])}) + {fmt_num(p['e'])} * np.sin({fmt_num(p['f'])} * x)",
        ),
        t(
            "L5_exp_sin_sq",
            "extreme",
            {
                "a": [1.0, 2.0, 3.0],
                "b": [0.5, 1.0, 2.0],
                "c": [1.0, 2.0, 3.0],
                "d": [0.5, 1.0],
                "e": [2.0, 3.0, 4.0],
            },
            [(-4.0, 4.0), (-6.0, 6.0)],
            ["np.exp", "np.sin", "np.cos"],
            lambda p: f"{fmt_num(p['a'])} * np.exp(-{fmt_num(p['b'])} * np.sin({fmt_num(p['c'])} * x) ** 2) + {fmt_num(p['d'])} * np.cos({fmt_num(p['e'])} * x)",
        ),
        t(
            "L5_fm_signal",
            "extreme",
            {"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0], "c": [1.0, 2.0, 3.0], "d": [1.0, 2.0, 3.0]},
            [(-4.0, 4.0), (-6.0, 6.0)],
            ["np.sin", "np.cos"],
            lambda p: f"{fmt_num(p['a'])} * np.sin({fmt_num(p['b'])} * x + {fmt_num(p['c'])} * np.cos({fmt_num(p['d'])} * x))",
        ),
        t(
            "L5_log_sin_sq_cos",
            "extreme",
            {
                "a": [1.0, 2.0],
                "b": [0.5, 1.0, 2.0],
                "c": [2.0, 4.0],
                "d": [0.5, 1.0],
                "e": [1.0, 2.0, 3.0],
            },
            [(-4.0, 4.0), (-5.0, 5.0)],
            ["np.log", "np.sin", "np.cos"],
            lambda p: f"np.log(np.abs({fmt_num(p['a'])} * np.sin({fmt_num(p['b'])} * x ** 2) + {fmt_num(p['c'])})) + {fmt_num(p['d'])} * np.cos({fmt_num(p['e'])} * x)",
        ),
    ]


def evaluate_expression(expr: str, x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        y = eval(expr, {"x": x, "np": np, "__builtins__": {}})
    y = np.asarray(y, dtype=np.float64)
    if y.shape == ():
        y = np.full_like(x, float(y))
    return y


def chebyshev_nodes(x_range: tuple[float, float], n: int) -> np.ndarray:
    lo, hi = x_range
    k = np.arange(n)
    nodes = np.cos((2 * k + 1) * math.pi / (2 * n))
    x = (lo + hi) / 2 + (hi - lo) / 2 * nodes
    return np.sort(x)


def make_data_points(expr: str, x_range: tuple[float, float], n: int) -> list[list[float]]:
    x = chebyshev_nodes(x_range, n)
    y = evaluate_expression(expr, x)
    if y.shape != x.shape or not np.all(np.isfinite(y)):
        raise ValueError(f"non-finite data points for {expr}")
    return [[float(a), float(b)] for a, b in zip(x, y)]


def make_test_points(expr: str, x_range: tuple[float, float], n: int) -> list[list[float]]:
    x = np.linspace(x_range[0], x_range[1], n)
    y = evaluate_expression(expr, x)
    if y.shape != x.shape or not np.all(np.isfinite(y)):
        raise ValueError(f"non-finite test points for {expr}")
    return [[float(a), float(b)] for a, b in zip(x, y)]


def render_plot(expr: str, x_range: tuple[float, float], image_path: Path) -> None:
    x = np.linspace(x_range[0], x_range[1], 900)
    y = evaluate_expression(expr, x)
    finite = np.isfinite(y)
    if not finite.any():
        raise ValueError(f"cannot render non-finite expression: {expr}")

    fig, ax = plt.subplots(figsize=(5.9, 3.9), dpi=100)
    ax.plot(x[finite], y[finite], color="#377eb8", linewidth=2.0, alpha=0.95)
    ax.set_xlim(x_range)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, color="#b0b0b0", linewidth=0.8, alpha=0.25)
    ax.axhline(0, color="#9e9e9e", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.axvline(0, color="#9e9e9e", linestyle="--", linewidth=0.6, alpha=0.55)
    fig.tight_layout()
    fig.savefig(image_path, format="png")
    plt.close(fig)


def build_prompt(function_hints: list[str], data_points: list[list[float]]) -> str:
    hints_text = (
        "Available functions: " + ", ".join(function_hints) + "\n"
        if function_hints
        else ""
    )
    points_text = "Reference points: " + "  ".join(
        f"({x:.4f}, {y:.4f})" for x, y in data_points
    ) + "\n"
    return DEFAULT_PROMPT.format(
        function_hints=hints_text,
        data_points=points_text,
        axis_note="",
    )


def build_tool_call(expr: str) -> str:
    payload = {
        "name": "submit_expression",
        "arguments": {"expression": expr},
    }
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def format_point_error(value: float) -> str:
    if not math.isfinite(value):
        return "invalid"
    if value < 1e-5:
        return "<1e-5"
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4g}"


def candidate_point_error(expr: str, data_points: list[list[float]]) -> float:
    x = np.asarray([point[0] for point in data_points], dtype=np.float64)
    y_true = np.asarray([point[1] for point in data_points], dtype=np.float64)
    try:
        y_pred = evaluate_expression(expr, x)
    except Exception:
        return float("inf")
    if y_pred.shape != y_true.shape or not np.all(np.isfinite(y_pred)):
        return float("inf")
    return float(np.max(np.abs(y_pred - y_true)))


def nearby_param_values(values: list[float], current: float) -> list[float]:
    unique_values = sorted(set(values))
    return [
        value
        for value in sorted(unique_values, key=lambda value: (abs(value - current), value))
        if abs(value - current) > 1e-12
    ]


def format_params(params: dict[str, float]) -> str:
    return ", ".join(f"{name}={fmt_num(value)}" for name, value in params.items())


def template_uses_allowed_functions(template: Template, function_hints: list[str]) -> bool:
    if not function_hints:
        return True
    return set(template.true_hints).issubset(set(function_hints))


def iter_param_guesses(
    template: Template,
    rng: random.Random,
    max_guesses: int,
) -> list[dict[str, float]]:
    names = list(template.param_choices)
    value_lists = [template.param_choices[name] for name in names]
    all_guesses = [
        {name: float(value) for name, value in zip(names, values)}
        for values in itertools.product(*value_lists)
    ]
    if max_guesses <= 0 or len(all_guesses) <= max_guesses:
        return all_guesses

    indices = list(range(len(all_guesses)))
    rng.shuffle(indices)
    selected = sorted(indices[:max_guesses])
    return [all_guesses[idx] for idx in selected]


def build_family_parameter_candidates(
    template: Template,
    data_points: list[list[float]],
    rng: random.Random,
    max_param_guesses: int,
    limit: int,
) -> list[dict]:
    candidates = []
    for params in iter_param_guesses(template, rng, max_param_guesses):
        expr = template.expression_builder(params)
        error = candidate_point_error(expr, data_points)
        if not math.isfinite(error):
            continue
        candidates.append(
            {
                "template_id": template.template_id,
                "expression": expr,
                "params": params,
                "max_abs_error": error,
            }
        )

    rng.shuffle(candidates)
    candidates.sort(key=lambda item: item["max_abs_error"])

    selected = []
    seen = set()
    for candidate in candidates:
        if candidate["expression"] in seen:
            continue
        selected.append(candidate)
        seen.add(candidate["expression"])
        if len(selected) >= limit:
            break
    return selected


def build_hard_negative_candidates(
    template: Template,
    params: dict[str, float],
    true_expr: str,
    data_points: list[list[float]],
    rng: random.Random,
    limit: int,
) -> list[dict]:
    if limit <= 0:
        return []

    candidates = []
    for name, values in template.param_choices.items():
        for alt_value in nearby_param_values(values, params[name])[:3]:
            alt_params = dict(params)
            alt_params[name] = alt_value
            alt_expr = template.expression_builder(alt_params)
            if alt_expr == true_expr:
                continue
            error = candidate_point_error(alt_expr, data_points)
            if not math.isfinite(error):
                continue
            candidates.append(
                {
                    "expression": alt_expr,
                    "changed_param": name,
                    "from": float(params[name]),
                    "to": float(alt_value),
                    "params": {key: float(value) for key, value in alt_params.items()},
                    "max_abs_error": error,
                }
            )

    # Shuffle before sorting so ties do not always expose the same parameter.
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: item["max_abs_error"])

    selected = []
    seen = {true_expr}
    for candidate in candidates:
        if candidate["expression"] in seen:
            continue
        selected.append(candidate)
        seen.add(candidate["expression"])
        if len(selected) >= limit:
            break
    return selected


FAMILY_DESCRIPTIONS = {
    "L1_sin": "a simple sinusoid, so nearby amplitude and frequency choices are plausible",
    "L1_cos": "a cosine wave, so frequency and amplitude are the main confusable parameters",
    "L1_exp_grow": "monotone exponential growth, so the scale and exponent coefficient need point checks",
    "L1_exp_decay": "monotone exponential decay, so nearby negative exponent coefficients are easy to confuse",
    "L1_sqrt": "a square-root curve, so scale factors are the main candidates",
    "L1_poly": "a symmetric/asymmetric power curve, so coefficient and power are the candidates",
    "L2_gaussian": "a Gaussian-like bump with an offset, so exp(-b*x**2) coefficients are plausible hard negatives",
    "L2_sin_plus_linear": "an oscillation riding on a line, so sine frequency and linear slope both need checking",
    "L2_sin_full": "a shifted sinusoid, so amplitude, frequency, phase, and vertical offset can all be confused",
    "L2_sin_cos": "a sine/cosine mixture, so the two frequencies are plausible alternatives",
    "L2_log": "a log-shaped even curve with an offset, so log scale and vertical shift need checking",
    "L2_exp_offset": "an exponential curve with a vertical offset, so exponent sign/size and offset are candidates",
    "L2_cos_full": "a shifted cosine, so adjacent frequencies such as 3 versus 4 must be tested",
    "L3_chirp": "a chirp-like oscillation where the phase depends on x**2",
    "L3_sqrt_sin": "an oscillation with a sqrt(abs(x)) envelope",
    "L3_damped_osc": "a damped cosine where decay rate and frequency are both visually plausible",
    "L3_gauss_sin": "a sinusoid under a Gaussian envelope, so envelope width and sine frequency are hard negatives",
    "L3_beat": "a beat pattern from multiplying sine and cosine terms",
    "L3_growing_osc": "an oscillation with an x envelope, so frequency and scale need reference-point checks",
    "L4_log_sin": "a nested log-sin expression, so log scale and sine frequency candidates are close",
    "L4_three_terms": "a three-term mixture of sinusoid, Gaussian bump, and linear trend",
    "L4_exp_chirp": "a decaying chirp, so decay and chirp coefficients are both candidates",
    "L4_sqrt_cos_sq": "a sqrt of a shifted cosine-of-x**2 curve",
    "L4_sin_of_exp": "a sine of an exponential phase, where exp coefficient and sine multiplier are easy to mix up",
    "L5_sqrt_chirp_poly": "a sqrt-chirp plus polynomial trend",
    "L5_tanh_nested": "a saturated tanh-sine structure plus another sinusoid",
    "L5_exp_sin_sq": "an exp(-b*sin(c*x)**2) envelope plus a cosine term",
    "L5_fm_signal": "a frequency-modulated sine where carrier and modulation frequencies are confusable",
    "L5_log_sin_sq_cos": "a log of a sin(x**2) term plus a cosine correction",
}


def build_candidate_family_rounds(
    all_templates: list[Template],
    true_template: Template,
    true_params: dict[str, float],
    true_expr: str,
    function_hints: list[str],
    data_points: list[list[float]],
    rng: random.Random,
    num_candidate_families: int,
    num_hard_negatives: int,
    max_family_param_guesses: int,
    accept_max_abs_error: float,
) -> list[dict]:
    if num_candidate_families < 1:
        num_candidate_families = 1

    wrong_options = []
    fallback_options = []
    for candidate_template in all_templates:
        if candidate_template.template_id == true_template.template_id:
            continue
        candidates = build_family_parameter_candidates(
            template=candidate_template,
            data_points=data_points,
            rng=rng,
            max_param_guesses=max_family_param_guesses,
            limit=max(2, num_hard_negatives),
        )
        if not candidates:
            continue
        option = {
            "template": candidate_template,
            "candidates": candidates,
            "best_error": candidates[0]["max_abs_error"],
        }
        if template_uses_allowed_functions(candidate_template, function_hints):
            wrong_options.append(option)
        else:
            fallback_options.append(option)

    # Prefer families that obey the prompt hints and can partially fit the
    # reference points, but avoid accepting an equivalent wrong-family expression
    # before the ground-truth family appears.
    def option_key(option: dict) -> tuple[float, float]:
        template = option["template"]
        return (option["best_error"], abs(template.difficulty - true_template.difficulty))

    wrong_options = [
        option for option in wrong_options if option["best_error"] > accept_max_abs_error
    ]
    wrong_options.sort(key=option_key)
    fallback_options = [
        option for option in fallback_options if option["best_error"] > accept_max_abs_error
    ]
    fallback_options.sort(key=option_key)

    selected_options = wrong_options[: max(0, num_candidate_families - 1)]
    if len(selected_options) < num_candidate_families - 1:
        needed = num_candidate_families - 1 - len(selected_options)
        selected_options.extend(fallback_options[:needed])

    rounds = []
    for option in selected_options:
        candidate_template = option["template"]
        rounds.append(
            {
                "template_id": candidate_template.template_id,
                "family_description": FAMILY_DESCRIPTIONS.get(
                    candidate_template.template_id,
                    "a function family consistent with the visible curve",
                ),
                "candidates": option["candidates"],
                "best_error": option["best_error"],
                "accepted": option["best_error"] <= accept_max_abs_error,
            }
        )

    hard_negatives = build_hard_negative_candidates(
        template=true_template,
        params=true_params,
        true_expr=true_expr,
        data_points=data_points,
        rng=rng,
        limit=num_hard_negatives,
    )
    true_candidate = {
        "template_id": true_template.template_id,
        "expression": true_expr,
        "changed_param": None,
        "from": None,
        "to": None,
        "params": {key: float(value) for key, value in true_params.items()},
        "max_abs_error": candidate_point_error(true_expr, data_points),
    }
    true_candidates = [true_candidate] + hard_negatives
    rounds.append(
        {
            "template_id": true_template.template_id,
            "family_description": FAMILY_DESCRIPTIONS.get(
                true_template.template_id,
                "the function family consistent with the visible curve",
            ),
            "candidates": true_candidates,
            "best_error": true_candidate["max_abs_error"],
            "accepted": true_candidate["max_abs_error"] <= accept_max_abs_error,
        }
    )
    return rounds


def build_point_check_answer(
    all_templates: list[Template],
    template: Template,
    params: dict[str, float],
    true_expr: str,
    function_hints: list[str],
    data_points: list[list[float]],
    rng: random.Random,
    num_hard_negatives: int,
    num_candidate_families: int,
    max_family_param_guesses: int,
    accept_max_abs_error: float,
) -> tuple[str, list[dict]]:
    family_rounds = build_candidate_family_rounds(
        all_templates=all_templates,
        true_template=template,
        true_params=params,
        true_expr=true_expr,
        function_hints=function_hints,
        data_points=data_points,
        rng=rng,
        num_candidate_families=num_candidate_families,
        num_hard_negatives=num_hard_negatives,
        max_family_param_guesses=max_family_param_guesses,
        accept_max_abs_error=accept_max_abs_error,
    )
    hint_text = ", ".join(function_hints) if function_hints else "polynomial terms"
    threshold_text = format_point_error(accept_max_abs_error)

    lines = [
        "<think>",
        f"The image and hints ({hint_text}) suggest several possible families, so I will test them in order.",
        f"Stopping rule: keep changing families while every candidate has max_abs_error above {threshold_text}.",
    ]
    accepted_round = None
    for round_idx, family_round in enumerate(family_rounds, start=1):
        lines.append(
            f"Family {round_idx}: {family_round['family_description']} "
            f"({family_round['template_id']})."
        )
        for candidate_idx, candidate in enumerate(family_round["candidates"], start=1):
            change = ""
            if candidate.get("changed_param") is not None:
                change = (
                    f" [{candidate['changed_param']}: "
                    f"{fmt_num(candidate['from'])}->{fmt_num(candidate['to'])}]"
                )
            lines.append(
                f"  guess {candidate_idx}: params({format_params(candidate['params'])}) -> "
                f"{candidate['expression']}{change}; "
                f"max_abs_error={format_point_error(candidate['max_abs_error'])}."
            )
        if family_round["best_error"] <= accept_max_abs_error:
            lines.append("  This family has a tiny reference-point error, so I stop searching.")
            accepted_round = family_round
            break
        lines.append("  All guesses in this family are still too far off, so I switch families.")

    if accepted_round is None:
        accepted_round = family_rounds[-1]

    lines.extend(
        [
            f"The expression with the smallest point error is {true_expr}, so I submit it.",
            "</think>",
            build_tool_call(true_expr),
        ]
    )
    return "\n".join(lines), family_rounds


def build_assistant_answer(
    assistant_style: str,
    all_templates: list[Template],
    template: Template,
    params: dict[str, float],
    true_expr: str,
    function_hints: list[str],
    data_points: list[list[float]],
    rng: random.Random,
    num_hard_negatives: int,
    num_candidate_families: int,
    max_family_param_guesses: int,
    accept_max_abs_error: float,
) -> tuple[str, list[dict]]:
    if assistant_style == "tool_only":
        return build_tool_call(true_expr), []
    if assistant_style == "point_check":
        return build_point_check_answer(
            all_templates=all_templates,
            template=template,
            params=params,
            true_expr=true_expr,
            function_hints=function_hints,
            data_points=data_points,
            rng=rng,
            num_hard_negatives=num_hard_negatives,
            num_candidate_families=num_candidate_families,
            max_family_param_guesses=max_family_param_guesses,
            accept_max_abs_error=accept_max_abs_error,
        )
    raise ValueError(f"unknown assistant style: {assistant_style}")


def sample_hints(template: Template, rng: random.Random) -> tuple[list[str], list[str]]:
    true_hints = list(template.true_hints)
    if template.difficulty_name == "easy":
        target_len = len(true_hints)
    elif template.difficulty_name == "medium":
        target_len = rng.randint(max(2, len(true_hints)), max(3, len(true_hints)))
    elif template.difficulty_name == "hard":
        target_len = rng.randint(max(3, len(true_hints)), 4)
    elif template.difficulty_name == "expert":
        target_len = rng.randint(max(4, len(true_hints)), 4)
    else:
        target_len = rng.randint(max(4, len(true_hints)), 6)

    distractor_pool = [f for f in ALL_FUNCTION_HINTS if f not in true_hints]
    distractors = rng.sample(distractor_pool, k=max(0, target_len - len(true_hints)))
    function_hints = true_hints + distractors
    rng.shuffle(function_hints)
    return true_hints, function_hints


def choose_template(
    templates: list[Template],
    rng: random.Random,
    forced_template_ids: list[str] | None,
) -> Template:
    if forced_template_ids:
        allowed = [t for t in templates if t.template_id in forced_template_ids]
        if not allowed:
            raise ValueError(f"unknown template ids: {forced_template_ids}")
        return rng.choice(allowed)
    return rng.choice(templates)


def build_template_schedule(
    templates: list[Template],
    rng: random.Random,
    num_samples: int,
    samples_per_template: int | None,
    forced_template_ids: list[str] | None,
    template_sample_counts: dict[str, int] | None = None,
) -> list[Template]:
    if template_sample_counts:
        template_by_id = {template.template_id: template for template in templates}
        unknown = sorted(set(template_sample_counts) - set(template_by_id))
        if unknown:
            raise ValueError(f"unknown template ids: {unknown}")
        schedule = []
        for template_id, count in template_sample_counts.items():
            if count < 1:
                raise ValueError(f"template sample count must be positive: {template_id}={count}")
            schedule.extend([template_by_id[template_id]] * count)
        rng.shuffle(schedule)
        return schedule

    if forced_template_ids:
        unknown = sorted(set(forced_template_ids) - {t.template_id for t in templates})
        if unknown:
            raise ValueError(f"unknown template ids: {unknown}")
        templates = [t for t in templates if t.template_id in set(forced_template_ids)]

    if samples_per_template is not None:
        schedule = [template for template in templates for _ in range(samples_per_template)]
    else:
        if num_samples < 1:
            raise ValueError("--num-samples must be positive")
        base, extra = divmod(num_samples, len(templates))
        schedule = []
        for idx, template in enumerate(templates):
            count = base + (1 if idx < extra else 0)
            schedule.extend([template] * count)

    rng.shuffle(schedule)
    return schedule


def make_sample(
    all_templates: list[Template],
    template: Template,
    sample_index: int,
    seed: int,
    out_dir: Path,
    rng: random.Random,
    min_points: int,
    max_points: int,
    n_test_points: int,
    assistant_style: str,
    num_hard_negatives: int,
    num_candidate_families: int,
    max_family_param_guesses: int,
    accept_max_abs_error: float,
) -> tuple[dict, dict]:
    params = {name: rng.choice(values) for name, values in template.param_choices.items()}
    expr = template.expression_builder(params)
    x_range = rng.choice(template.x_ranges)
    n_points = rng.randint(min_points, max_points)
    sample_id = f"synth_{template.difficulty_name}_{template.template_id}_{sample_index:07d}"
    image_rel = f"images/{sample_id}.png"
    image_abs = out_dir / image_rel

    data_points = make_data_points(expr, x_range, n_points)
    test_points = make_test_points(expr, x_range, n_test_points)
    render_plot(expr, x_range, image_abs)

    true_hints, function_hints = sample_hints(template, rng)
    prompt = build_prompt(function_hints, data_points)
    assistant_answer, candidate_checks = build_assistant_answer(
        assistant_style=assistant_style,
        all_templates=all_templates,
        template=template,
        params=params,
        true_expr=expr,
        function_hints=function_hints,
        data_points=data_points,
        rng=rng,
        num_hard_negatives=num_hard_negatives,
        num_candidate_families=num_candidate_families,
        max_family_param_guesses=max_family_param_guesses,
        accept_max_abs_error=accept_max_abs_error,
    )
    expression_str = expr.replace("np.", "")

    task_sample = {
        "id": sample_id,
        "split": "synth_train",
        "ood_type": None,
        "ood_subtype": None,
        "difficulty": template.difficulty,
        "difficulty_name": template.difficulty_name,
        "expression_str": expression_str,
        "expression_numpy": expr,
        "true_function_hints": true_hints,
        "function_hints": function_hints,
        "data_points_text": data_points,
        "data_points_image": [],
        "data_points_all": data_points,
        "n_text_points": len(data_points),
        "n_image_points": 0,
        "image_path": image_rel,
        "image_x_range": [float(x_range[0]), float(x_range[1])],
        "y_axis_log_scale": False,
        "has_data_point_annotations": False,
        "test_points": test_points,
        "test_x_in_range": [float(p[0]) for p in test_points],
        "test_x_out_range": [],
        "generation_seed": seed,
        "generation_config": {
            "template_id": template.template_id,
            "sampled_params": params,
            "x_sample_method": "chebyshev",
            "n_data_points_total": len(data_points),
            "n_test_points": n_test_points,
            "distractor_funcs": [f for f in function_hints if f not in true_hints],
            "full_function_domain": [float(x_range[0]), float(x_range[1])],
            "assistant_style": assistant_style,
            "num_candidate_families": num_candidate_families,
            "max_family_param_guesses": max_family_param_guesses,
            "accept_max_abs_error": accept_max_abs_error,
            "candidate_checks": candidate_checks,
        },
    }

    sft_sample = {
        "id": sample_id,
        "image": image_rel,
        "template_id": template.template_id,
        "difficulty_name": template.difficulty_name,
        "expression_numpy": expr,
        "assistant_style": assistant_style,
        "num_candidate_families": num_candidate_families,
        "accept_max_abs_error": accept_max_abs_error,
        "candidate_checks": candidate_checks,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rel},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": assistant_answer},
        ],
    }
    return task_sample, sft_sample


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(
    task_rows: list[dict],
    sft_rows: list[dict],
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")

    indices = list(range(len(task_rows)))
    rng.shuffle(indices)
    n_val = int(round(len(indices) * val_ratio))
    val_indices = set(indices[:n_val])

    task_train: list[dict] = []
    task_val: list[dict] = []
    sft_train: list[dict] = []
    sft_val: list[dict] = []
    for idx, (task_row, sft_row) in enumerate(zip(task_rows, sft_rows)):
        if idx in val_indices:
            task_row["split"] = "synth_val"
            task_val.append(task_row)
            sft_val.append(sft_row)
        else:
            task_row["split"] = "synth_train"
            task_train.append(task_row)
            sft_train.append(sft_row)
    return task_train, task_val, sft_train, sft_val


def parse_template_sample_counts(spec: str) -> dict[str, int] | None:
    if not spec.strip():
        return None
    counts: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid --template-samples item: {item!r}; expected TEMPLATE_ID=COUNT")
        template_id, raw_count = item.split("=", 1)
        template_id = template_id.strip()
        try:
            count = int(raw_count.strip())
        except ValueError as exc:
            raise ValueError(f"invalid sample count for {template_id!r}: {raw_count!r}") from exc
        if not template_id:
            raise ValueError(f"invalid empty template id in --template-samples item: {item!r}")
        if count < 1:
            raise ValueError(f"template sample count must be positive: {template_id}={count}")
        counts[template_id] = count
    return counts or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stage-1 SFT data")
    parser.add_argument("--out", type=Path, default=Path("data/stage1_synth"))
    parser.add_argument("--num-samples", type=int, default=2900)
    parser.add_argument(
        "--samples-per-template",
        type=int,
        default=None,
        help="Generate this many samples for each selected template. Overrides --num-samples.",
    )
    parser.add_argument(
        "--template-samples",
        type=str,
        default="",
        help=(
            "Comma-separated TEMPLATE_ID=COUNT entries. When set, overrides "
            "--num-samples, --samples-per-template, and --templates."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--max-points", type=int, default=20)
    parser.add_argument("--n-test-points", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument(
        "--assistant-style",
        choices=["tool_only", "point_check"],
        default="tool_only",
        help=(
            "Assistant target style. point_check adds short reasoning that "
            "guesses candidate families from the image/hints, tests hard-negative "
            "parameter variants on reference points, then calls the tool."
        ),
    )
    parser.add_argument(
        "--num-hard-negatives",
        type=int,
        default=3,
        help="Number of nearby wrong parameter candidates in point_check targets.",
    )
    parser.add_argument(
        "--num-candidate-families",
        type=int,
        default=3,
        help="Number of candidate family rounds in point_check targets, including the accepted family.",
    )
    parser.add_argument(
        "--max-family-param-guesses",
        type=int,
        default=512,
        help="Maximum parameter guesses to score per candidate family.",
    )
    parser.add_argument(
        "--accept-max-abs-error",
        type=float,
        default=1e-4,
        help="Reference-point max absolute error threshold for accepting a candidate family.",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="",
        help="Comma-separated template ids to generate from; default uses all 29.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_points < 2 or args.max_points < args.min_points:
        raise SystemExit("--min-points/--max-points must define a valid interval")
    if args.samples_per_template is not None and args.samples_per_template < 1:
        raise SystemExit("--samples-per-template must be positive")
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be in [0, 1)")
    if args.num_hard_negatives < 0:
        raise SystemExit("--num-hard-negatives must be non-negative")
    if args.num_candidate_families < 1:
        raise SystemExit("--num-candidate-families must be positive")
    if args.max_family_param_guesses < 1:
        raise SystemExit("--max-family-param-guesses must be positive")
    if args.accept_max_abs_error <= 0:
        raise SystemExit("--accept-max-abs-error must be positive")
    try:
        template_sample_counts = parse_template_sample_counts(args.template_samples)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "images").mkdir(parents=True, exist_ok=True)

    templates = build_templates()
    forced_template_ids = [s.strip() for s in args.templates.split(",") if s.strip()] or None
    rng = random.Random(args.seed)
    split_rng = random.Random(args.seed + 17)
    schedule = build_template_schedule(
        templates=templates,
        rng=rng,
        num_samples=args.num_samples,
        samples_per_template=args.samples_per_template,
        forced_template_ids=forced_template_ids,
        template_sample_counts=template_sample_counts,
    )

    task_rows: list[dict] = []
    sft_rows: list[dict] = []
    for i, template in enumerate(schedule):
        task_sample, sft_sample = make_sample(
            all_templates=templates,
            template=template,
            sample_index=i,
            seed=args.seed,
            out_dir=args.out,
            rng=rng,
            min_points=args.min_points,
            max_points=args.max_points,
            n_test_points=args.n_test_points,
            assistant_style=args.assistant_style,
            num_hard_negatives=args.num_hard_negatives,
            num_candidate_families=args.num_candidate_families,
            max_family_param_guesses=args.max_family_param_guesses,
            accept_max_abs_error=args.accept_max_abs_error,
        )
        task_rows.append(task_sample)
        sft_rows.append(sft_sample)
        if (i + 1) % 100 == 0 or i + 1 == len(schedule):
            print(f"[{i + 1}/{len(schedule)}] generated", flush=True)

    task_train, task_val, sft_train, sft_val = split_rows(
        task_rows=task_rows,
        sft_rows=sft_rows,
        val_ratio=args.val_ratio,
        rng=split_rng,
    )

    write_jsonl(args.out / "samples.jsonl", task_rows)
    write_jsonl(args.out / "sft_messages.jsonl", sft_rows)
    write_jsonl(args.out / "samples_train.jsonl", task_train)
    write_jsonl(args.out / "samples_val.jsonl", task_val)
    write_jsonl(args.out / "sft_train.jsonl", sft_train)
    write_jsonl(args.out / "sft_val.jsonl", sft_val)

    template_counts = collections.Counter(row["generation_config"]["template_id"] for row in task_rows)
    difficulty_counts = collections.Counter(row["difficulty_name"] for row in task_rows)

    manifest = {
        "num_samples": len(task_rows),
        "num_train": len(task_train),
        "num_val": len(task_val),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "assistant_style": args.assistant_style,
        "num_hard_negatives": args.num_hard_negatives,
        "num_candidate_families": args.num_candidate_families,
        "max_family_param_guesses": args.max_family_param_guesses,
        "accept_max_abs_error": args.accept_max_abs_error,
        "template_samples_arg": args.template_samples,
        "num_templates": len(template_counts),
        "templates": sorted(template_counts),
        "template_counts": dict(sorted(template_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "outputs": {
            "task_samples": "samples.jsonl",
            "sft_messages": "sft_messages.jsonl",
            "task_train": "samples_train.jsonl",
            "task_val": "samples_val.jsonl",
            "sft_train": "sft_train.jsonl",
            "sft_val": "sft_val.jsonl",
            "images": "images/",
        },
        "prompt_template": DEFAULT_PROMPT,
    }
    with (args.out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(task_rows)} samples to {args.out}")
    print(f"Task-format JSONL: {args.out / 'samples.jsonl'}")
    print(f"SFT messages JSONL: {args.out / 'sft_messages.jsonl'}")
    print(f"SFT train JSONL: {args.out / 'sft_train.jsonl'}")
    print(f"SFT val JSONL: {args.out / 'sft_val.jsonl'}")


if __name__ == "__main__":
    main()
