#!/usr/bin/env bash

# Run a single MT-Bench sample with expert trace collection for MoE models.

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
PYTHON_ABS=$(command -v "$PYTHON_BIN")
BASE_CMD="${PYTHON_ABS} eagle/evaluation/eval_eagle.py"

CUDA_VISIBLE_DEVICES_BASE=${CUDA_VISIBLE_DEVICES_BASE:-0}
CUDA_VISIBLE_DEVICES_OVERRIDE=${CUDA_VISIBLE_DEVICES_OVERRIDE:-0,1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES_BASE}

LOG_DIR="results/mt_bench/logs"
mkdir -p "$LOG_DIR"

declare -a MOE_MODELS=(
  "allenai/OLMoE-1B-7B-0125-Instruct|wantsleep/OLMoE_1B_7B_Eagle3|olmoe-1b-trace"
  "Qwen/Qwen3-30B-A3B|Tengyunw/qwen3_30b_moe_eagle3|qwen3-30b-trace"
)

for entry in "${MOE_MODELS[@]}"; do
  IFS='|' read -r BASE_MODEL EA_MODEL MODEL_ID <<<"$entry"
  LOG_PATH="$LOG_DIR/${MODEL_ID}.log"
  if [[ "$MODEL_ID" == "qwen3-30b-trace" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_OVERRIDE"
  else
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_BASE"
  fi
  echo "==> Collecting expert traces for $MODEL_ID"
  $BASE_CMD \
    --base-model-path "$BASE_MODEL" \
    --ea-model-path "$EA_MODEL" \
    --model-id "$MODEL_ID" \
    --bench-name mt_bench \
    --num-questions 1 \
    --warmup-tokens 16 \
    --max-new-tokens 1024 \
    --temperature 1.0 \
    --use-eagle \
    --use-eagle3 \
    --collect-expert-traces \
    >"$LOG_PATH" 2>&1
done

echo "Logs written to $LOG_DIR"
