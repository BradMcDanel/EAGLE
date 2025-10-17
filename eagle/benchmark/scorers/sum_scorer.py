from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _load_questions(question_file: Path) -> Dict[int, Dict[str, Any]]:
    data: Dict[int, Dict[str, Any]] = {}
    with question_file.open() as fh:
        for line in fh:
            record = json.loads(line)
            data[int(record["question_id"])] = record
    return data


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            temp = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def _rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    lcs = _lcs_length(pred_tokens, ref_tokens)

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def score(
    answer_file: Path,
    question_file: Path,
    output_path: Path,
    run_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    questions = _load_questions(question_file)
    details = []

    with answer_file.open() as fh:
        for line in fh:
            record = json.loads(line)
            qid = int(record["question_id"])
            question = questions[qid]
            reference_list = question.get("reference") or []
            reference = " ".join(reference_list).strip()

            turns: List[str] = record["choices"][0]["turns"]
            prediction = turns[-1].strip() if turns else ""

            rouge = _rouge_l(prediction, reference)
            stats = record["choices"][0].get("stats", [])
            stat = stats[-1] if stats else {}

            details.append(
                {
                    "question_id": qid,
                    "reference": reference,
                    "prediction": prediction,
                    "rouge_l": rouge,
                    "tokens": stat.get("tokens"),
                    "throughput": stat.get("throughput"),
                    "wall_time": stat.get("time"),
                }
            )

    if details:
        avg_precision = sum(item["rouge_l"]["precision"] for item in details) / len(details)
        avg_recall = sum(item["rouge_l"]["recall"] for item in details) / len(details)
        avg_f1 = sum(item["rouge_l"]["f1"] for item in details) / len(details)
    else:
        avg_precision = avg_recall = avg_f1 = 0.0

    metrics = {
        "bench_name": run_metadata["bench_name"],
        "model_id": run_metadata["model_id"],
        "variant": run_metadata["variant"],
        "answer_file": str(answer_file),
        "question_file": str(question_file),
        "num_questions": len(details),
        "rouge_l": {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        },
        "details": details,
        "generation_stats": run_metadata.get("generation_stats", {}),
        "config": run_metadata.get("config", {}),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics

