#!/usr/bin/env bash
# Offline RL/DPO runtime for HW3. Source this file before launching training.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
SHARED_ENV="${SHARED_ENV:-${PUBLIC_ROOT}/envs/hw3}"
USER_GLOBAL="${USER_GLOBAL:-/inspire/hdd/global_user/zhongxiaoqiu-253108120179}"

if [[ ! -f "${SHARED_ENV}/bin/activate" ]]; then
  echo "Shared hw3 env not found: ${SHARED_ENV}" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${SHARED_ENV}/bin/activate"

export PROJECT_ROOT
export PUBLIC_ROOT
export HF_HOME="${HF_HOME:-${USER_GLOBAL}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_GLOBAL}/.cache/pip}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PIP_NO_INDEX="${PIP_NO_INDEX:-1}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_NO_TORCHAUDIO="${TRANSFORMERS_NO_TORCHAUDIO:-1}"
export TRANSFORMERS_NO_AUDIO="${TRANSFORMERS_NO_AUDIO:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"
echo "Activated offline RL env: ${SHARED_ENV}"
