#!/usr/bin/env bash

# Run MT-Bench (2 questions) for a set of models with and without EAGLE,
# optionally sweeping expert budgets for MoE models. Produces per-run logs
# under results/mt_bench/logs and prints a throughput summary table.

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
PYTHON_ABS=$(command -v "$PYTHON_BIN")
BASE_CMD="${PYTHON_ABS} eagle/evaluation/eval_eagle.py"

CUDA_VISIBLE_DEVICES_BASE=${CUDA_VISIBLE_DEVICES_BASE:-0}
CUDA_VISIBLE_DEVICES_OVERRIDE=${CUDA_VISIBLE_DEVICES_OVERRIDE:-0,1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES_BASE}

LOG_DIR="results/mt_bench/logs"
mkdir -p "$LOG_DIR"

declare -a MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct|yuhuili/EAGLE3-LLaMA3.1-Instruct-8B|llama31-8b|"
  "Qwen/Qwen3-8B|Tengyunw/qwen3_8b_eagle3|qwen3-8b|"
  "allenai/OLMoE-1B-7B-0125-Instruct|wantsleep/OLMoE_1B_7B_Eagle3|olmoe-1b|48,32,16"
  "Qwen/Qwen3-30B-A3B|Tengyunw/qwen3_30b_moe_eagle3|qwen3-30b|48,32,16"
)

run_baseline() {
  local base_model=$1
  local ea_model=$2
  local model_id=$3
  local log_path=$4

  if [[ "$model_id" == "qwen3-30b" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_OVERRIDE"
  else
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_BASE"
  fi

  if [[ -s "$log_path" ]]; then
    echo "==> Baseline: $model_id (skipping, log exists)"
    return
  fi

  echo "==> Baseline: $model_id"
  echo "$BASE_CMD --base-model-path \"$base_model\" --ea-model-path \"$ea_model\" --model-id \"${model_id}-baseline\" --bench-name mt_bench --num-questions 2 --warmup-tokens 16 --max-new-tokens 1024 --temperature 0.0 --use-eagle3"
  $BASE_CMD \
    --base-model-path "$base_model" \
    --ea-model-path "$ea_model" \
    --model-id "${model_id}-baseline" \
    --bench-name mt_bench \
    --num-questions 2 \
    --warmup-tokens 16 \
    --max-new-tokens 1024 \
    --temperature 0.0 \
    --use-eagle3 \
    >"$log_path" 2>&1
}

run_eagle() {
  local base_model=$1
  local ea_model=$2
  local model_id=$3
  local suffix=$4
  local extra=$5
  local log_path=$6

  if [[ "$model_id" == "qwen3-30b" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_OVERRIDE"
  else
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_BASE"
  fi

  if [[ -s "$log_path" ]]; then
    echo "==> EAGLE${suffix:+ ($suffix)}: $model_id (skipping, log exists)"
    return
  fi

  echo "==> EAGLE${suffix:+ ($suffix)}: $model_id"
  echo "$BASE_CMD --base-model-path \"$base_model\" --ea-model-path \"$ea_model\" --model-id \"${model_id}-eagle${suffix}\" --bench-name mt_bench --num-questions 2 --warmup-tokens 16 --max-new-tokens 1024 --temperature 0.0 --use-eagle --use-eagle3 ${extra}"
  $BASE_CMD \
    --base-model-path "$base_model" \
    --ea-model-path "$ea_model" \
    --model-id "${model_id}-eagle${suffix}" \
    --bench-name mt_bench \
    --num-questions 2 \
    --warmup-tokens 16 \
    --max-new-tokens 1024 \
    --temperature 0.0 \
    --use-eagle \
    --use-eagle3 \
    $extra \
    >"$log_path" 2>&1
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r BASE_MODEL EA_MODEL SHORT_ID BUDGETS <<<"$entry"

  BASE_LOG="$LOG_DIR/${SHORT_ID}_baseline.log"
  run_baseline "$BASE_MODEL" "$EA_MODEL" "$SHORT_ID" "$BASE_LOG"

  EAGLE_LOG="$LOG_DIR/${SHORT_ID}_eagle.log"
  run_eagle "$BASE_MODEL" "$EA_MODEL" "$SHORT_ID" "" "" "$EAGLE_LOG"

  if [[ -n "$BUDGETS" ]]; then
    BASE_MOE_LOG="$LOG_DIR/${SHORT_ID}_eagle_B0.log"
    run_eagle "$BASE_MODEL" "$EA_MODEL" "$SHORT_ID" "_B0" "" "$BASE_MOE_LOG"
    IFS=',' read -ra BUDGET_LIST <<<"$BUDGETS"
    for budget in "${BUDGET_LIST[@]}"; do
      SUFFIX="_B${budget}"
      LOG="$LOG_DIR/${SHORT_ID}_eagle${SUFFIX}.log"
      run_eagle "$BASE_MODEL" "$EA_MODEL" "$SHORT_ID" "$SUFFIX" "--max-active-experts ${budget}" "$LOG"
    done
  fi
done

${PYTHON_BIN:-python} scripts/summarize_mt_bench_logs.py
