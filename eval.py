#!/usr/bin/env python3
"""
eval.py -- Symbolic regression evaluation (IND only).

Usage:
    python eval.py <model_checkpoint_path>
    python eval.py <model_checkpoint_path> --split dev
    python eval.py <model_checkpoint_path> --split test

Results saved to:
    <script_dir>/eval_outputs/<model_name>/eval_results[_<split>].jsonl
    <script_dir>/eval_outputs/<model_name>/eval_summary[_<split>].json

Prompt override:
    Place prompt.txt in <model_checkpoint_path>/ to replace the default template.
    Placeholders: {function_hints}  {data_points}  {axis_note}
"""

import argparse
import base64
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ==============================================================================
# Constants
# ==============================================================================

DATA_DIR       = Path(__file__).parent / "data"
BATCH_SIZE     = 4096
MAX_NEW_TOKENS = 16384
MAX_MODEL_LEN  = 32768
N_TEST_MAX     = 50

# Tool definition passed to the model
TOOL_SUBMIT = {
    "type": "function",
    "function": {
        "name": "submit_expression",
        "description": "Submit the inferred numpy expression for f(x).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Numpy expression for f(x), e.g. '2 * np.sin(3 * x + 1)'",
                }
            },
            "required": ["expression"],
        },
    },
}

DEFAULT_PROMPT = """\
You are a symbolic regression expert. Given a function curve plot and reference \
information, call submit_expression with the inferred numpy expression.

{function_hints}{data_points}{axis_note}
You may reason inside <think>...</think> before calling the tool.
Use only numpy (np.) functions; the variable must be named x."""

# ==============================================================================
# Data loading
# ==============================================================================

def load_samples(data_dir: Path) -> list:
    samples = []
    for path in sorted(data_dir.rglob("samples.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    s["_data_dir"] = str(path.parent)
                    samples.append(s)
    return samples

# ==============================================================================
# Message building
# ==============================================================================

def build_message(sample: dict, template: str, b64_image: str) -> list:
    """Return an OpenAI-style messages list for one sample."""
    hints = sample.get("function_hints", [])
    function_hints = ("Available functions: " + ", ".join(hints) + "\n") if hints else ""

    pts = sample.get("data_points_text", [])
    if pts:
        data_points = "Reference points: " + "  ".join(f"({x:.4f}, {y:.4f})" for x, y in pts) + "\n"
    else:
        data_points = ""

    axis_note = ""

    text = template.format(
        function_hints=function_hints,
        data_points=data_points,
        axis_note=axis_note,
    )

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                {"type": "text", "text": text},
            ],
        }
    ]

# ==============================================================================
# Expression extraction  (uses vLLM's own parsers when available)
# ==============================================================================

class _MockRequest:
    """Minimal stand-in for ChatCompletionRequest expected by vLLM parsers."""
    def __init__(self):
        self.tools = None


_MOCK_REQUEST = _MockRequest()


def build_parsers(reasoning_parser: Optional[str],
                  tool_call_parser: Optional[str],
                  tokenizer) -> dict:
    """Instantiate vLLM reasoning / tool-call parsers. Returns a dict."""
    parsers: dict = {"reasoning": None, "tool_call": None}

    if reasoning_parser:
        from vllm.reasoning import ReasoningParserManager
        cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser)
        parsers["reasoning"] = cls(tokenizer)
        print(f"[INFO] Reasoning parser: {reasoning_parser}")

    if tool_call_parser:
        try:
            from vllm.tool_parsers import ToolParserManager
        except ImportError:
            from vllm.entrypoints.openai.tool_parsers import ToolParserManager
        cls = ToolParserManager.get_tool_parser(tool_call_parser)
        parsers["tool_call"] = cls(tokenizer)
        print(f"[INFO] Tool-call parser: {tool_call_parser}")

    return parsers


def extract_expression(raw_text: str, parsers: dict) -> Optional[str]:
    """Extract expression from raw model output text."""
    if not raw_text:
        return None

    # --- Step 1: strip reasoning / thinking ---
    content = raw_text
    rp = parsers.get("reasoning")
    if rp is not None:
        extract_fn = getattr(rp, "extract_reasoning_content",
                             getattr(rp, "extract_reasoning", None))
        _, content = extract_fn(raw_text, request=_MOCK_REQUEST)
        if content is None:
            content = raw_text

    # --- Step 2: extract tool call ---
    tp = parsers.get("tool_call")
    if tp is not None:
        try:
            info = tp.extract_tool_calls(content, request=_MOCK_REQUEST)
            if info.tools_called:
                for tc in info.tool_calls:
                    args = json.loads(tc.function.arguments)
                    expr = args.get("expression", "").strip()
                    if expr:
                        return expr
        except Exception:
            pass

    # --- Step 3: regex fallback ---
    if rp is None:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    if tp is None:
        m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
        if m:
            expr = _expr_from_json(m.group(1))
            if expr:
                return expr

    for m in re.finditer(r'\{[^{}]*"expression"\s*:\s*"([^"]+)"[^{}]*\}', content):
        expr = m.group(1).strip()
        if expr:
            return expr

    return None


def _expr_from_json(json_str: str) -> Optional[str]:
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data.get("arguments"), dict):
        return data["arguments"].get("expression", "").strip() or None
    if "expression" in data:
        return str(data["expression"]).strip() or None
    return None

# ==============================================================================
# Metrics computation
# ==============================================================================

R2_THRESHOLDS = [0.99, 0.95, 0.90, 0.80]


def compute_metrics(expr: str, test_points: list) -> dict:
    """Return {"mse": float|None, "r2": float|None}."""
    fail = {"mse": None, "r2": None}
    if not expr or not test_points:
        return fail
    x = np.array([p[0] for p in test_points[:N_TEST_MAX]])
    y = np.array([p[1] for p in test_points[:N_TEST_MAX]])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = eval(expr, {"x": x, "np": np, "__builtins__": {}})
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.shape == ():
            y_pred = np.full_like(x, float(y_pred))
        if y_pred.shape != x.shape or not np.all(np.isfinite(y_pred)):
            return fail
    except Exception:
        return fail

    ss_res = float(np.sum((y_pred - y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    mse = ss_res / len(y)

    if ss_tot < 1e-12:
        r2 = 1.0 if ss_res < 1e-8 else -1.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return {"mse": round(mse, 6), "r2": round(r2, 6)}

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Symbolic regression evaluation (IND only)")
    parser.add_argument("model_path", help="Path to the model checkpoint directory")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size (default: 1)")
    parser.add_argument("--dp", type=int, default=1,
                        help="Data parallel size (default: 1)")
    parser.add_argument("--reasoning-parser", type=str, default="qwen3",
                        help="vLLM reasoning parser name, e.g. qwen3, deepseek_r1")
    parser.add_argument("--tool-call-parser", type=str, default="hermes",
                        help="vLLM tool-call parser name, e.g. hermes, mistral, llama, pythonic")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Skip torch.compile to avoid SHM timeout on first launch")
    parser.add_argument("--no-v1", action="store_true",
                        help="Disable vLLM V1 engine (set VLLM_USE_V1=0)")
    parser.add_argument("--split", type=str, default=None,
                        choices=["dev", "test"],
                        help="Evaluate a single split from data/task/ (default: all data)")
    args = parser.parse_args()

    if args.no_v1:
        import os
        os.environ["VLLM_USE_V1"] = "0"
        print("[INFO] vLLM V1 engine disabled (VLLM_USE_V1=0)")

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print(f"[ERROR] Model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    prompt_file = model_path / "prompt.txt"
    template = prompt_file.read_text(encoding="utf-8").strip() if prompt_file.exists() else DEFAULT_PROMPT
    print(f"[INFO] Prompt: {'custom' if prompt_file.exists() else 'default'}")

    if args.split:
        data_dir = Path(__file__).parent / "data" / "task" / args.split
    else:
        data_dir = DATA_DIR

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    samples = load_samples(data_dir)
    print(f"[INFO] Loaded {len(samples)} samples"
          + (f" (split={args.split})" if args.split else ""))

    print(f"[INFO] Loading model from {model_path} ...")
    print(f"[INFO] Parallelism: tp={args.tp}  dp={args.dp}")
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(e)
        print("[ERROR] vllm not installed. Run: pip install vllm", file=sys.stderr)
        sys.exit(1)

    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=args.tp,
        data_parallel_size=args.dp,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=args.enforce_eager,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)
    print("[INFO] Model loaded")

    tokenizer = llm.get_tokenizer()
    parsers = build_parsers(args.reasoning_parser, args.tool_call_parser, tokenizer)

    results = []
    t0 = time.time()

    for batch_start in range(0, len(samples), BATCH_SIZE):
        batch         = samples[batch_start : batch_start + BATCH_SIZE]
        conversations = []
        valid_indices = []

        for idx, sample in enumerate(batch):
            img_path = Path(sample["_data_dir"]) / sample["image_path"]
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                results.append({
                    "id": sample["id"],
                    "mse": None, "r2": None,
                })
                continue
            conversations.append(build_message(sample, template, b64))
            valid_indices.append(idx)

        if not conversations:
            continue

        try:
            outputs = llm.chat(
                conversations,
                tools=[TOOL_SUBMIT],
                sampling_params=sampling_params,
            )
        except Exception as exc:
            print(f"[WARN] Batch failed: {exc}")
            for idx in valid_indices:
                s = batch[idx]
                results.append({
                    "id": s["id"],
                    "mse": None, "r2": None,
                })
            continue

        for idx, output_obj in zip(valid_indices, outputs):
            sample = batch[idx]
            raw_text = output_obj.outputs[0].text or ""
            expr   = extract_expression(raw_text, parsers)
            metrics = compute_metrics(expr, sample.get("test_points", []))
            results.append({
                "id":              sample["id"],
                "true_expr":       sample["expression_numpy"],
                "predicted_expr":  expr,
                "mse":             metrics["mse"],
                "r2":              metrics["r2"],
            })

        n_done  = min(batch_start + BATCH_SIZE, len(samples))
        elapsed = time.time() - t0
        print(f"  [{n_done}/{len(samples)}]  {elapsed:.0f}s  ETA={((len(samples)-n_done)/max(n_done/elapsed,1e-9)):.0f}s")


    # Summary: overall average accuracy (R²-based)
    total = len(results)
    r2s = [r["r2"] for r in results]
    n_null = sum(1 for v in r2s if v is None)
    valid_r2s = [v for v in r2s if v is not None]

    accs = {}
    for t in R2_THRESHOLDS:
        n_pass = sum(1 for v in r2s if v is not None and v >= t)
        accs[t] = n_pass / total if total > 0 else 0.0

    med_r2 = float(np.median(valid_r2s)) if valid_r2s else float("nan")
    mean_r2 = float(np.mean(valid_r2s)) if valid_r2s else float("nan")

    summary = {
        "total":     total,
        "null":      n_null,
        **{f"acc@{t}": round(accs[t], 4) for t in R2_THRESHOLDS},
        "mean_r2":   round(mean_r2, 6) if np.isfinite(mean_r2) else None,
        "median_r2": round(med_r2, 6) if np.isfinite(med_r2) else None,
    }

    tau_labels = "  ".join(f"{'acc@'+str(t):>8}" for t in R2_THRESHOLDS)
    print(f"\n{'total':>5}  {tau_labels}  {'null%':>6}  {'mean_R2':>8}  {'med_R2':>8}")
    print("-" * 70)
    acc_str = "  ".join(f"{accs[t]:>8.1%}" for t in R2_THRESHOLDS)
    print(f"{total:>5}  {acc_str}  {n_null/max(total,1):>5.1%}  {mean_r2:>8.4f}  {med_r2:>8.4f}")
    
    suffix = f"_{args.split}" if args.split else ""
    script_dir = Path(__file__).parent
    output_dir = script_dir / "eval_outputs" / model_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"eval_results{suffix}.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[INFO] Per-sample results -> {results_path}")

    summary_path = output_dir / f"eval_summary{suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
