from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GSM8KResult:
    question_id: int
    reference: str
    prediction: str
    correct: bool
    tokens: Optional[int]
    throughput: Optional[float]
    wall_time: Optional[float]


def _load_questions(question_file: Path) -> Dict[int, Dict[str, Any]]:
    data: Dict[int, Dict[str, Any]] = {}
    with question_file.open() as fh:
        for line in fh:
            record = json.loads(line)
            data[int(record["question_id"])] = record
    return data


def _extract_reference(question: Dict[str, Any]) -> str:
    refs = question.get("reference") or []
    for ref in refs:
        if "####" in ref:
            return ref.split("####")[-1].strip()
    if refs:
        return str(refs[0]).strip()
    raise ValueError(f"Reference answer missing for question {question['question_id']}")


def _extract_prediction(text: str) -> str:
    if "####" in text:
        candidate = text.split("####")[-1].strip()
        if candidate:
            return candidate

    cleaned = text
    cleaned = cleaned.replace("\\boxed", "")
    cleaned = cleaned.replace("\\(", "").replace("\\)", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("{", " ").replace("}", " ")

    # Last numeric value in the text
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if matches:
        return matches[-1]

    tokens = [tok.strip(" ,.;:\n") for tok in cleaned.replace("\n", " ").split(" ") if tok]
    return tokens[-1] if tokens else ""


def _normalize_number(text: str) -> Optional[str]:
    stripped = text.replace(",", "").strip()
    if not stripped:
        return None
    try:
        if "." in stripped:
            value = float(stripped)
            return ("%f" % value).rstrip("0").rstrip(".")
        return str(int(stripped))
    except ValueError:
        return stripped.lower()


def score(
    answer_file: Path,
    question_file: Path,
    output_path: Path,
    run_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    questions = _load_questions(question_file)
    details: List[GSM8KResult] = []

    with answer_file.open() as fh:
        for line in fh:
            record = json.loads(line)
            qid = int(record["question_id"])
            question = questions[qid]
            reference = _extract_reference(question)

            turns: List[str] = record["choices"][0]["turns"]
            prediction = _extract_prediction(turns[-1]) if turns else ""

            stats = record["choices"][0].get("stats", [])
            stat = stats[-1] if stats else {}

            ref_norm = _normalize_number(reference)
            pred_norm = _normalize_number(prediction)
            correct = bool(ref_norm is not None and pred_norm == ref_norm)

            details.append(
                GSM8KResult(
                    question_id=qid,
                    reference=reference,
                    prediction=prediction,
                    correct=correct,
                    tokens=stat.get("tokens"),
                    throughput=stat.get("throughput"),
                    wall_time=stat.get("time"),
                )
            )

    total = len(details)
    accuracy = sum(1 for item in details if item.correct) / total if total else 0.0

    metrics = {
        "bench_name": run_metadata["bench_name"],
        "model_id": run_metadata["model_id"],
        "variant": run_metadata["variant"],
        "answer_file": str(answer_file),
        "question_file": str(question_file),
        "num_questions": total,
        "accuracy": accuracy,
        "correct": sum(1 for item in details if item.correct),
        "details": [
            {
                "question_id": item.question_id,
                "reference": item.reference,
                "prediction": item.prediction,
                "correct": item.correct,
                "tokens": item.tokens,
                "throughput": item.throughput,
                "wall_time": item.wall_time,
            }
            for item in details
        ],
        "generation_stats": run_metadata.get("generation_stats", {}),
        "config": run_metadata.get("config", {}),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics
