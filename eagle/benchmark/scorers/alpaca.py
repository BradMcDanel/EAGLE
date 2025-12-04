"""Scorer for the Alpaca instruction-following benchmark."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _f1_score(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = pred_counter & ref_counter
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, reference: str) -> float:
    return float(_normalize_text(prediction) == _normalize_text(reference))


def _load_questions(question_file: Path) -> Dict[int, Dict[str, Any]]:
    questions: Dict[int, Dict[str, Any]] = {}
    with question_file.open() as fh:
        for line in fh:
            record = json.loads(line)
            questions[int(record["question_id"])] = record
    return questions


def score(
    answer_file: Path,
    question_file: Path,
    output_path: Path,
    run_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    questions = _load_questions(question_file)

    details: List[Dict[str, Any]] = []
    with answer_file.open() as fh:
        for line in fh:
            record = json.loads(line)
            qid = int(record["question_id"])
            assert qid in questions, f"Question {qid} not found in question file"
            question = questions[qid]
            references = question["reference"]

            turns: List[str] = record["choices"][0]["turns"]
            assert len(turns) > 0, f"No turns found for question {qid}"
            prediction = turns[-1]

            stats = record["choices"][0].get("stats", [])
            stat = stats[-1] if len(stats) > 0 else {}

            if references:
                em_scores = [_exact_match(prediction, ref) for ref in references]
                f1_scores = [_f1_score(prediction, ref) for ref in references]
                exact = max(em_scores)
                f1 = max(f1_scores)
            else:
                exact = 0.0
                f1 = 0.0

            details.append(
                {
                    "question_id": qid,
                    "prediction": prediction,
                    "references": references,
                    "exact_match": exact,
                    "f1": f1,
                    "tokens": stat.get("tokens"),
                    "throughput": stat.get("throughput"),
                    "wall_time": stat.get("time"),
                }
            )

    num_questions = len(details)
    avg_em = sum(item["exact_match"] for item in details) / num_questions if num_questions else 0.0
    avg_f1 = sum(item["f1"] for item in details) / num_questions if num_questions else 0.0

    metrics = {
        "bench_name": run_metadata["bench_name"],
        "model_id": run_metadata["model_id"],
        "variant": run_metadata["variant"],
        "answer_file": str(answer_file),
        "question_file": str(question_file),
        "num_questions": num_questions,
        "exact_match": avg_em,
        "f1": avg_f1,
        "details": details,
        "generation_stats": run_metadata.get("generation_stats", {}),
        "config": run_metadata.get("config", {}),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics
