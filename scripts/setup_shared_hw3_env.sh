#!/usr/bin/env bash
set -euo pipefail

PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
USER_GLOBAL="${USER_GLOBAL:-/inspire/hdd/global_user/zhongxiaoqiu-253108120179}"
ENV_PREFIX="${ENV_PREFIX:-${PUBLIC_ROOT}/envs/hw3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"

export HF_HOME="${HF_HOME:-${USER_GLOBAL}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_GLOBAL}/.cache/pip}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"

mkdir -p "$PUBLIC_ROOT/envs" "$HF_HOME" "$HF_HUB_CACHE" "$PIP_CACHE_DIR"

if [[ ! -d "$ENV_PREFIX" ]]; then
  if command -v conda >/dev/null 2>&1; then
    conda create -y -p "$ENV_PREFIX" "python=${PYTHON_VERSION}" pip
  else
    python3 -m venv "$ENV_PREFIX"
  fi
fi

# shellcheck disable=SC1091
source "$ENV_PREFIX/bin/activate"

python -m pip install -U pip setuptools wheel

if [[ "$INSTALL_TORCH" == "1" ]]; then
  python -m pip install --extra-index-url "$TORCH_INDEX_URL" torch torchvision
fi

python -m pip install -U \
  "git+https://github.com/huggingface/transformers" \
  accelerate \
  peft \
  bitsandbytes \
  pillow \
  tensorboard \
  safetensors \
  sentencepiece \
  protobuf \
  qwen-vl-utils

python - <<'PY'
import accelerate
import peft
import torch
import transformers

print("python environment ready")
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("accelerate", accelerate.__version__)
from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
print("Qwen3VL import ok")
PY

chmod -R a+rwX "$ENV_PREFIX"

echo
echo "Shared environment ready at:"
echo "  $ENV_PREFIX"
echo
echo "Activate from either online or offline instance with:"
echo "  source $ENV_PREFIX/bin/activate"
