#!/usr/bin/env python3
"""Utilities for verifier-guided symbolic-regression RL experiments."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


FUNC_RE = re.compile(r"np\.([A-Za-z_][A-Za-z0-9_]*)")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?")
SAFE_NP_NAMES = {
    "sin",
    "cos",
    "exp",
    "sqrt",
    "log",
    "abs",
    "tanh",
    "tan",
    "arctan",
    "arccos",
    "arcsin",
    "sinh",
    "cosh",
}


@dataclass(frozen=True)
class MetricResult:
    mse: float | None
    r2: float | None
    max_abs_error: float | None
    valid: bool


class SafeExpressionVisitor(ast.NodeVisitor):
    """Conservative AST allowlist for numpy expressions in this task."""

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Attribute,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
    )

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self.allowed_nodes):
            raise ValueError(f"disallowed expression node: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in {"x", "np"}:
            raise ValueError(f"disallowed name: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not isinstance(node.value, ast.Name) or node.value.id != "np":
            raise ValueError("only np.<function> attributes are allowed")
        if node.attr not in SAFE_NP_NAMES:
            raise ValueError(f"disallowed numpy function: {node.attr}")


def validate_expression(expr: str) -> bool:
    if not expr or len(expr) > 512:
        return False
    try:
        tree = ast.parse(expr, mode="eval")
        SafeExpressionVisitor().visit(tree)
    except Exception:
        return False
    return True


def evaluate_expression(expr: str, x: np.ndarray) -> np.ndarray | None:
    if not validate_expression(expr):
        return None
    try:
        with np.errstate(all="ignore"):
            y = eval(expr, {"x": x, "np": np, "__builtins__": {}})
        y = np.asarray(y, dtype=np.float64)
        if y.shape == ():
            y = np.full_like(x, float(y))
        if y.shape != x.shape or not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None


def compute_metrics(expr: str, points: list[list[float]], max_points: int | None = None) -> MetricResult:
    if not expr or not points:
        return MetricResult(None, None, None, False)
    used_points = points[:max_points] if max_points else points
    x = np.asarray([point[0] for point in used_points], dtype=np.float64)
    y_true = np.asarray([point[1] for point in used_points], dtype=np.float64)
    y_pred = evaluate_expression(expr, x)
    if y_pred is None:
        return MetricResult(None, None, None, False)

    residual = y_pred - y_true
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    mse = ss_res / max(len(y_true), 1)
    if ss_tot < 1e-12:
        r2 = 1.0 if ss_res < 1e-8 else -1.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    max_abs_error = float(np.max(np.abs(residual)))
    return MetricResult(mse, r2, max_abs_error, True)


def shaped_reward(metrics: MetricResult) -> float:
    if not metrics.valid or metrics.r2 is None or metrics.mse is None:
        return -1.0
    clipped_r2 = max(-1.0, min(1.0, float(metrics.r2)))
    pass_bonus = 1.0 if metrics.r2 >= 0.99 else 0.0
    near_bonus = 0.35 if metrics.r2 >= 0.95 else 0.0
    ok_bonus = 0.15 if metrics.r2 >= 0.80 else 0.0
    return clipped_r2 + pass_bonus + near_bonus + ok_bonus


def expression_functions(expr: str) -> set[str]:
    return set(FUNC_RE.findall(expr or ""))


def expression_numbers(expr: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_RE.finditer(expr or ""):
        try:
            values.append(float(match.group(0)))
        except ValueError:
            pass
    return values


def replace_one_number(expr: str, index: int, new_value: float) -> str:
    matches = list(NUMBER_RE.finditer(expr))
    if index < 0 or index >= len(matches):
        return expr
    match = matches[index]
    if abs(new_value - round(new_value)) < 1e-10:
        formatted = str(int(round(new_value)))
    else:
        formatted = f"{new_value:.6g}"
    return expr[: match.start()] + formatted + expr[match.end() :]


def mutate_expression_numbers(expr: str, max_mutations: int = 10) -> list[str]:
    numbers = expression_numbers(expr)
    candidates: list[str] = []
    seen = {expr}
    for idx, value in enumerate(numbers):
        alternatives = [
            value + 1.0,
            value - 1.0,
            value * 2.0,
            value * 0.5,
            -value,
        ]
        for alt in alternatives:
            if abs(alt) < 1e-12 and abs(value) < 1e-12:
                continue
            mutated = replace_one_number(expr, idx, alt)
            if mutated not in seen and validate_expression(mutated):
                candidates.append(mutated)
                seen.add(mutated)
                if len(candidates) >= max_mutations:
                    return candidates
    return candidates


def generic_distractors(function_hints: list[str]) -> list[str]:
    funcs = {hint.replace("np.", "") for hint in function_hints}
    distractors = [
        "1 * x",
        "1 * x ** 2",
        "1 * x ** 3",
        "1 * np.sin(1 * x)",
        "2 * np.sin(3 * x)",
        "1 * np.cos(2 * x)",
        "2 * np.exp(-0.5 * x ** 2)",
        "1 * np.log(np.abs(1 * x) + 1)",
        "1 * np.sqrt(np.abs(x))",
    ]
    if "sin" in funcs and "cos" not in funcs:
        distractors.append("1 * np.cos(1 * x)")
    if "exp" in funcs:
        distractors.append("1 * np.exp(1 * x)")
    return distractors


def build_candidate_features(
    expr: str,
    reference_points: list[list[float]],
    function_hints: list[str],
) -> list[float]:
    ref_metrics = compute_metrics(expr, reference_points)
    funcs = expression_functions(expr)
    hints = {hint.replace("np.", "") for hint in function_hints}
    hint_overlap = len(funcs & hints) / max(len(funcs | hints), 1)
    ref_mse = ref_metrics.mse if ref_metrics.mse is not None else 1e6
    ref_max = ref_metrics.max_abs_error if ref_metrics.max_abs_error is not None else 1e6
    ref_r2 = ref_metrics.r2 if ref_metrics.r2 is not None else -1.0
    return [
        1.0,
        -math.log1p(max(ref_mse, 0.0)),
        -math.log1p(max(ref_max, 0.0)),
        max(-1.0, min(1.0, ref_r2)),
        hint_overlap,
        -min(len(expr), 300) / 300.0,
    ]


def summarize_jsonable_metrics(metrics: MetricResult) -> dict[str, Any]:
    return {
        "mse": metrics.mse,
        "r2": metrics.r2,
        "max_abs_error": metrics.max_abs_error,
        "valid": metrics.valid,
    }
