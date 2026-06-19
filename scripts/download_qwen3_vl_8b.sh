#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
USER_GLOBAL="${USER_GLOBAL:-/inspire/hdd/global_user/zhongxiaoqiu-253108120179}"
MODEL_DIR="${MODEL_DIR:-${PUBLIC_ROOT}/models/Qwen3-VL-8B-Instruct}"
HF_HOME="${HF_HOME:-${USER_GLOBAL}/.cache/huggingface}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_GLOBAL}/.cache/pip}"
MAX_WORKERS="${MAX_WORKERS:-8}"

export MODEL_ID
export PUBLIC_ROOT
export USER_GLOBAL
export MODEL_DIR
export MAX_WORKERS
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PIP_CACHE_DIR
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

mkdir -p "$MODEL_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$PIP_CACHE_DIR"

python3 - <<'PY'
import importlib.util
import subprocess
import sys

missing = [
    pkg for pkg, mod in {
        "huggingface_hub": "huggingface_hub",
        "hf_transfer": "hf_transfer",
    }.items()
    if importlib.util.find_spec(mod) is None
]
if missing:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        *missing,
    ])
PY

python3 - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
model_dir = Path(os.environ["MODEL_DIR"])
cache_dir = os.environ["HF_HUB_CACHE"]
max_workers = int(os.environ["MAX_WORKERS"])

print(f"Downloading {model_id}")
print(f"Model dir: {model_dir}")
print(f"HF cache:  {cache_dir}")

snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    cache_dir=cache_dir,
    max_workers=max_workers,
)

config = model_dir / "config.json"
safetensors = sorted(model_dir.glob("*.safetensors"))
if not config.exists():
    raise SystemExit(f"Missing {config}")
if not safetensors:
    raise SystemExit(f"No safetensors found in {model_dir}")

print(f"Downloaded files into {model_dir}")
print(f"Safetensors shards: {len(safetensors)}")
PY

chmod -R a+rX "$MODEL_DIR"
find "$MODEL_DIR" -maxdepth 1 -type f | sort | sed -n '1,40p'
du -sh "$MODEL_DIR"
