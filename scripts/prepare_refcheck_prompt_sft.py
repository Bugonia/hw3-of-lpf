#!/usr/bin/env python3
"""Create an SFT copy whose user prompt includes a short reference-point check."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_REFCHECK_LINE = (
    "Before submitting, privately substitute your candidate into several Reference points; "
    "if the computed values do not match, revise the formula instead of calling the tool."
)

ANCHOR_LINE = "You may reason inside <think>...</think> before calling the tool."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def insert_refcheck_line(text: str, line: str) -> str:
    if line in text:
        return text
    if ANCHOR_LINE in text:
        return text.replace(ANCHOR_LINE, line + "\n" + ANCHOR_LINE)
    return text.rstrip() + "\n" + line


def rewrite_messages(row: dict[str, Any], line: str) -> dict[str, Any]:
    row = json.loads(json.dumps(row, ensure_ascii=False))
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return row

    content = messages[0].get("content")
    if not isinstance(content, list):
        return row

    changed = False
    for item in content:
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            item["text"] = insert_refcheck_line(item["text"], line)
            changed = True
            break
    if not changed:
        content.append({"type": "text", "text": line})
    row["prompt_refcheck"] = True
    return row


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists; pass --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def link_images(src_dir: Path, out_dir: Path) -> None:
    src_images = src_dir / "images"
    if not src_images.exists():
        return
    dst_images = out_dir / "images"
    os.symlink(src_images.resolve(), dst_images)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copyfile(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Source SFT data directory")
    parser.add_argument("--out", type=Path, required=True, help="Output SFT data directory")
    parser.add_argument("--line", default=DEFAULT_REFCHECK_LINE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_dir = args.src
    out_dir = args.out
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)

    train_in = src_dir / "sft_train.jsonl"
    val_in = src_dir / "sft_val.jsonl"
    if not train_in.exists() or not val_in.exists():
        raise FileNotFoundError("source directory must contain sft_train.jsonl and sft_val.jsonl")

    prepare_output_dir(out_dir, args.overwrite)
    link_images(src_dir, out_dir)

    counts: dict[str, int] = {}
    for name in ("sft_train.jsonl", "sft_val.jsonl", "sft_messages.jsonl"):
        src_path = src_dir / name
        if not src_path.exists():
            continue
        rows = [rewrite_messages(row, args.line) for row in read_jsonl(src_path)]
        write_jsonl(out_dir / name, rows)
        counts[name] = len(rows)

    for name in ("samples.jsonl", "samples_train.jsonl", "samples_val.jsonl"):
        copy_if_exists(src_dir / name, out_dir / name)

    prompt_template = None
    manifest_path = src_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prompt_template = manifest.get("prompt_template")

    if isinstance(prompt_template, str):
        prompt_text = insert_refcheck_line(prompt_template.strip(), args.line)
    else:
        first_row = read_jsonl(train_in)[0]
        prompt_text = ""
        for item in first_row["messages"][0]["content"]:
            if item.get("type") == "text":
                prompt_text = insert_refcheck_line(item["text"].strip(), args.line)
                break
    (out_dir / "prompt.txt").write_text(prompt_text + "\n", encoding="utf-8")

    out_manifest = {
        "source_dir": str(src_dir),
        "refcheck_line": args.line,
        "counts": counts,
        "image_dir": "images",
        "source_manifest": manifest,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(out_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
