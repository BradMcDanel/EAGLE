#!/usr/bin/env bash

# Judge MT-Bench answers for baseline vs. EAGLE variants using FastChat (v0.2.31 API).
# Creates a temporary FastChat-style data directory per model pair, runs the
# pairwise-baseline judge, and writes raw judgments plus a JSON summary.

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4}
PARALLEL=${PARALLEL:-1}
MAX_QUESTIONS=${MAX_QUESTIONS:-2}          # Use 2 to match the current subset
TEMPERATURE_TAG=${TEMPERATURE_TAG:-t0.0}   # Suffix used in filename matching
RESULTS_DIR=${RESULTS_DIR:-results/mt_bench}
QUESTION_FILE=${QUESTION_FILE:-eagle/data/mt_bench/question.jsonl}
REFERENCE_DIR=${REFERENCE_DIR:-eagle/data/mt_bench/reference_answer}
JUDGE_FILE=${JUDGE_FILE:-eagle/data/judge_prompts.jsonl}
FASTCHAT_STAGE_ROOT=${FASTCHAT_STAGE_ROOT:-${RESULTS_DIR}/fastchat_stage}
JUDGMENT_OUTPUT_DIR=${JUDGMENT_OUTPUT_DIR:-${RESULTS_DIR}/judgments}
OVERWRITE=${OVERWRITE:-0}

mkdir -p "${JUDGMENT_OUTPUT_DIR}"
mkdir -p "${FASTCHAT_STAGE_ROOT}/data"

# Baseline / variant pairs to evaluate
# Each entry is BASELINE:VARIANT (model_id values inside the JSONL answer files)
declare -a MODEL_PAIRS=(
  "llama31-8b-baseline:llama31-8b-eagle"
  "qwen3-8b-baseline:qwen3-8b-eagle"
  "olmoe-1b-baseline:olmoe-1b-eagle"
  "olmoe-1b-baseline:olmoe-1b-eagle_B0"
  "olmoe-1b-baseline:olmoe-1b-eagle_B16"
  "olmoe-1b-baseline:olmoe-1b-eagle_B32"
  "olmoe-1b-baseline:olmoe-1b-eagle_B48"
  "qwen3-30b-baseline:qwen3-30b-eagle"
  "qwen3-30b-baseline:qwen3-30b-eagle_B0"
  "qwen3-30b-baseline:qwen3-30b-eagle_B16"
  "qwen3-30b-baseline:qwen3-30b-eagle_B32"
  "qwen3-30b-baseline:qwen3-30b-eagle_B48"
)

realpath_f() {
  python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

# Ensure FastChat can find the judge prompt file at its default location
ln -sf "$(realpath_f "${JUDGE_FILE}")" "${FASTCHAT_STAGE_ROOT}/data/judge_prompts.jsonl"

locate_answer_file() {
  local model_id="$1"
  local pattern="${RESULTS_DIR}/${model_id}-*-${TEMPERATURE_TAG}.jsonl"
  local file
  file=$(ls ${pattern} 2>/dev/null | head -n1 || true)
  if [[ -z "$file" ]]; then
    echo ""; return 1
  fi
  echo "$file"
}

for pair in "${MODEL_PAIRS[@]}"; do
  IFS=':' read -r baseline_id variant_id <<<"${pair}"
  slug="${baseline_id}_vs_${variant_id}"
  bench_name="mt_bench_${slug}"
  bench_dir="${FASTCHAT_STAGE_ROOT}/data/${bench_name}"
  model_answer_dir="${bench_dir}/model_answer"
  ref_dir="${bench_dir}/reference_answer"
  model_judgment_dir="${bench_dir}/model_judgment"

  judgment_dest="${JUDGMENT_OUTPUT_DIR}/${slug}.jsonl"
  summary_dest="${JUDGMENT_OUTPUT_DIR}/${slug}_summary.json"

  if [[ -s "${judgment_dest}" && ${OVERWRITE} -ne 1 ]]; then
    echo "==> Skipping ${slug} (judgment already exists)"
    continue
  fi

  echo "==> Preparing data for ${slug}"
  mkdir -p "${model_answer_dir}" "${ref_dir}" "${model_judgment_dir}"

  # Ensure clean model answers for this pair
  find "${model_answer_dir}" -type l -delete 2>/dev/null || true
  find "${model_answer_dir}" -type f -delete 2>/dev/null || true

  # Link question file
  ln -sf "$(realpath_f "${QUESTION_FILE}")" "${bench_dir}/question.jsonl"

  # Link reference answers
  for ref_file in "${REFERENCE_DIR}"/*.jsonl; do
    ln -sf "$(realpath_f "${ref_file}")" "${ref_dir}/$(basename "${ref_file}")"
  done

  # Link baseline answer
  baseline_src=$(locate_answer_file "${baseline_id}") || true
  if [[ -z "${baseline_src}" ]]; then
    echo "[error] Could not find answer file for ${baseline_id}" >&2
    exit 1
  fi
  ln -sf "$(realpath_f "${baseline_src}")" "${model_answer_dir}/${baseline_id}.jsonl"

  # Link variant answer
  variant_src=$(locate_answer_file "${variant_id}") || true
  if [[ -z "${variant_src}" ]]; then
    echo "[error] Could not find answer file for ${variant_id}" >&2
    exit 1
  fi
  ln -sf "$(realpath_f "${variant_src}")" "${model_answer_dir}/${variant_id}.jsonl"

  echo "==> Judging ${slug}"
  pushd "${FASTCHAT_STAGE_ROOT}" >/dev/null
  printf '\n' | ${PYTHON_BIN} -m fastchat.llm_judge.gen_judgment \
    --bench-name "${bench_name}" \
    --judge-model "${JUDGE_MODEL}" \
    --baseline-model "${baseline_id}" \
    --mode pairwise-baseline \
    --model-list "${variant_id}" \
    --parallel "${PARALLEL}" \
    --first-n "${MAX_QUESTIONS}" >/tmp/mtbench_judge_${slug}.log 2>&1 || {
      status=$?
      popd >/dev/null
      echo "[error] gen_judgment failed for ${slug}; see /tmp/mtbench_judge_${slug}.log" >&2
      cat /tmp/mtbench_judge_${slug}.log >&2
      exit ${status}
    }
  popd >/dev/null

  raw_judgment_file="${model_judgment_dir}/${JUDGE_MODEL}_pair.jsonl"
  if [[ ! -s "${raw_judgment_file}" ]]; then
    echo "[error] Expected judgment file not found for ${slug}" >&2
    exit 1
  fi

  cp "${raw_judgment_file}" "${judgment_dest}"

  echo "==> Summarizing ${slug}"
  ${PYTHON_BIN} - <<'PY' "${judgment_dest}" "${baseline_id}" "${variant_id}" "${bench_name}" "${summary_dest}"
import json, sys, pathlib
judgment_path = pathlib.Path(sys.argv[1])
baseline = sys.argv[2]
variant = sys.argv[3]
bench_name = sys.argv[4]
summary_path = pathlib.Path(sys.argv[5])

wins = losses = ties = total = 0
question_ids = set()

with judgment_path.open() as f:
    for line in f:
        rec = json.loads(line)
        g1, g2 = rec.get("g1_winner"), rec.get("g2_winner")
        if g1 == "error" or g2 == "error":
            continue
        models = {rec.get("model_1"), rec.get("model_2")}
        if baseline not in models or variant not in models:
            continue
        total += 1
        question_ids.add(rec.get("question_id"))
        if g1 == "tie" or g1 != g2:
            ties += 1
            continue
        winner = rec["model_1"] if g1 == "model_1" else rec["model_2"]
        if winner == variant:
            wins += 1
        else:
            losses += 1

matches = wins + losses + ties
summary = {
    "bench_name": bench_name,
    "baseline_model": baseline,
    "variant_model": variant,
    "judgment_file": str(judgment_path),
    "questions_evaluated": len(question_ids),
    "matches": matches,
    "wins": wins,
    "losses": losses,
    "ties": ties,
}
if matches:
    summary["win_rate"] = wins / matches
    summary["loss_rate"] = losses / matches
    summary["tie_rate"] = ties / matches
    summary["win_rate_adjusted"] = (wins + 0.5 * ties) / matches
with summary_path.open("w") as out:
    json.dump(summary, out, indent=2)
PY

  echo "==> Stored judgment: ${judgment_dest}"
  echo "==> Stored summary:  ${summary_dest}"

done

echo "All judgments complete. Results under ${JUDGMENT_OUTPUT_DIR}."
