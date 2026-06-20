#!/usr/bin/env bash
set -euo pipefail

PUBLIC_ROOT="${PUBLIC_ROOT:-/inspire/hdd/project/generative-large-model/public}"
PROJECT_ROOT="${PROJECT_ROOT:-${PUBLIC_ROOT}/hw3-of-lpf}"
export ADAPTER_DIR="${ADAPTER_DIR:-${PROJECT_ROOT}/outputs/qwen3_vl_stage5_reasoning_lora}"
export MERGED_DIR="${MERGED_DIR:-${PROJECT_ROOT}/outputs/qwen3_vl_stage5_reasoning_merged}"

cd "$PROJECT_ROOT"
bash scripts/merge_stage1_lora.sh
