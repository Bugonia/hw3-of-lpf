#!/usr/bin/env python3
"""Generate stage-1 symbolic-regression SFT data.

The generator mirrors the released dev distribution: each sample contains a
curve image, function hints with distractors, Chebyshev-sampled reference
points, the ground-truth numpy expression, and an assistant tool-call answer.
"""

from __future__ import annotations

import argparse
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


def make_sample(
    template: Template,
    sample_index: int,
    seed: int,
    out_dir: Path,
    rng: random.Random,
    min_points: int,
    max_points: int,
    n_test_points: int,
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
        },
    }

    sft_sample = {
        "id": sample_id,
        "image": image_rel,
        "template_id": template.template_id,
        "difficulty_name": template.difficulty_name,
        "expression_numpy": expr,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rel},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": build_tool_call(expr)},
        ],
    }
    return task_sample, sft_sample


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stage-1 SFT data")
    parser.add_argument("--out", type=Path, default=Path("data/stage1_synth"))
    parser.add_argument("--num-samples", type=int, default=2900)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--max-points", type=int, default=20)
    parser.add_argument("--n-test-points", type=int, default=50)
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

    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "images").mkdir(parents=True, exist_ok=True)

    templates = build_templates()
    forced_template_ids = [s.strip() for s in args.templates.split(",") if s.strip()] or None
    rng = random.Random(args.seed)

    task_rows: list[dict] = []
    sft_rows: list[dict] = []
    for i in range(args.num_samples):
        template = choose_template(templates, rng, forced_template_ids)
        task_sample, sft_sample = make_sample(
            template=template,
            sample_index=i,
            seed=args.seed,
            out_dir=args.out,
            rng=rng,
            min_points=args.min_points,
            max_points=args.max_points,
            n_test_points=args.n_test_points,
        )
        task_rows.append(task_sample)
        sft_rows.append(sft_sample)

    write_jsonl(args.out / "samples.jsonl", task_rows)
    write_jsonl(args.out / "sft_messages.jsonl", sft_rows)

    manifest = {
        "num_samples": len(task_rows),
        "seed": args.seed,
        "num_templates": len(templates),
        "templates": [t.template_id for t in templates],
        "outputs": {
            "task_samples": "samples.jsonl",
            "sft_messages": "sft_messages.jsonl",
            "images": "images/",
        },
        "prompt_template": DEFAULT_PROMPT,
    }
    with (args.out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(task_rows)} samples to {args.out}")
    print(f"Task-format JSONL: {args.out / 'samples.jsonl'}")
    print(f"SFT messages JSONL: {args.out / 'sft_messages.jsonl'}")


if __name__ == "__main__":
    main()
