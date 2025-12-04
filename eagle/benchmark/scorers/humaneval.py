from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _ensure_human_eval_import() -> None:
    """Make sure the local human-eval checkout is importable."""

    try:
        import human_eval  # type: ignore  # noqa: F401
    except ImportError:
        repo_root = Path(__file__).resolve().parents[3]
        human_eval_repo = repo_root / "human-eval"
        if human_eval_repo.exists():
            sys.path.insert(0, str(human_eval_repo))
        # Re-raise if the import still fails
        import human_eval  # type: ignore  # noqa: F401


_ensure_human_eval_import()

from human_eval.data import HUMAN_EVAL, read_problems  # type: ignore


@dataclass
class _AnswerRecord:
    question_id: int
    sample: Dict[str, Any]
    raw_text: str


def _trim_to_entry_point(code: str, entry_point: Optional[str]) -> str:
    """Trim a snippet so it starts at the target entry point definition, if present."""

    if not code:
        return ""

    if entry_point:
        pattern = re.compile(rf"\bdef\s+{re.escape(entry_point)}\s*\(")
        match = pattern.search(code)
        if match:
            return code[match.start():]

    return code


def _extract_code(text: str, entry_point: Optional[str] = None) -> str:
    """Return a code snippet from a model answer."""

    if not text:
        return ""

    selected = ""
    candidates = [block.strip() for block in _CODE_FENCE.findall(text) if block.strip()]
    entry_pattern: Optional[re.Pattern[str]] = None
    if entry_point:
        entry_pattern = re.compile(rf"\bdef\s+{re.escape(entry_point)}\s*\(")

    if entry_pattern:
        for candidate in candidates:
            if entry_pattern.search(candidate):
                selected = candidate
                break

    if not selected:
        for candidate in candidates:
            if "def " in candidate:
                selected = candidate
                break

    if not selected and candidates:
        selected = candidates[0]

    if not selected:
        selected = text.strip()

    return _trim_to_entry_point(selected.strip(), entry_point)


def _normalize_completion(code: str) -> str:
    """Format the completion so it can be appended to the HumanEval prompt."""

    if not code:
        return ""

    lines = code.strip().splitlines()

    header: Optional[str] = None
    body_lines: List[str]

    if lines and lines[0].lstrip().startswith("def "):
        header = lines[0].lstrip()
        raw_body = lines[1:]
        trimmed_body: List[str] = []
        for line in raw_body:
            if not line.strip():
                trimmed_body.append("")
                continue
            if not line.startswith((" ", "\t")):
                break
            trimmed_body.append(line.rstrip())
        body_lines = trimmed_body
    else:
        body_lines = lines

    if header is not None and not body_lines:
        return ""

    if header is None:
        body_lines = [("    " + line) if line.strip() else "" for line in body_lines]
    else:
        body_lines = [line if line.startswith(("    ", "\t")) else f"    {line.lstrip()}" for line in body_lines]

    body = "\n".join(body_lines).rstrip()
    if not body.endswith("\n"):
        body += "\n"
    return body


def _load_questions(question_file: Path) -> Dict[int, str]:
    """Load questions and extract function names to map question_id -> HumanEval task_id."""
    question_to_function = {}
    with question_file.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            entry = json.loads(line)
            question_id = int(entry["question_id"])
            # Extract function name from the prompt
            prompt = entry["turns"][0] if "turns" in entry else ""
            # Look for 'def function_name(' - use findall to get all, take last
            # (HumanEval puts helper functions first, target function last)
            matches = re.findall(r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', prompt)
            if matches:
                function_name = matches[-1]  # Take the last function definition
                question_to_function[question_id] = function_name
    return question_to_function


def _load_answers(answer_file: Path, question_to_task: Dict[int, str]) -> List[_AnswerRecord]:
    """Load answers and map to HumanEval task_ids using question mapping."""
    records: List[_AnswerRecord] = []
    with answer_file.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            entry = json.loads(line)
            question_id = int(entry["question_id"])
            turns = entry["choices"][0]["turns"]
            assert len(turns) > 0, f"No turns found for question {question_id}"
            raw_text = turns[-1]

            # Get the task_id from the mapping
            assert question_id in question_to_task, \
                f"Question {question_id} not found in question file mapping"
            task_id = question_to_task[question_id]

            records.append(
                _AnswerRecord(
                    question_id=question_id,
                    raw_text=raw_text,
                    sample={
                        "task_id": task_id,
                        "completion": "",
                    },
                )
            )
    return records


def _write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")


def _run_completion(problem: Dict[str, Any], completion: str, timeout: float) -> Dict[str, Any]:
    check_program = (
        problem["prompt"]
        + completion
        + "\n"
        + problem["test"]
        + "\n"
        + f"check({problem['entry_point']})\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "check.py"
        script_path.write_text(check_program)

        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if completed.returncode == 0:
                result = "passed"
            else:
                stderr = completed.stderr.strip()
                stdout = completed.stdout.strip()
                result = f"failed: {stderr or stdout or 'tests failed'}"
        except subprocess.TimeoutExpired:
            result = "timed out"

    return {"task_id": problem["task_id"], "passed": result == "passed", "result": result}


def score(
    answer_file: Path,
    question_file: Path,
    output_path: Path,
    run_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    # Load questions to map question_id -> function name -> HumanEval task_id
    question_to_function = _load_questions(question_file)

    # Load all HumanEval problems and create reverse mapping: function name -> task_id
    all_problems = read_problems(HUMAN_EVAL)
    function_to_task = {p["entry_point"]: tid for tid, p in all_problems.items()}

    # Create mapping: question_id -> HumanEval task_id
    question_to_task = {}
    for qid, func_name in question_to_function.items():
        assert func_name in function_to_task, \
            f"Function '{func_name}' from question {qid} not found in HumanEval dataset"
        question_to_task[qid] = function_to_task[func_name]

    # Load answers with proper task_id mapping
    answers = _load_answers(answer_file, question_to_task)
    if not answers:
        raise ValueError(f"No answers found in {answer_file}")

    target_ids = {ans.sample["task_id"] for ans in answers}

    missing = sorted(task_id for task_id in target_ids if task_id not in all_problems)
    if missing:
        raise ValueError(f"Missing HumanEval tasks: {', '.join(missing)}")

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        samples_path = tmpdir / "samples.jsonl"
        problems_path = tmpdir / "subset_problems.jsonl"

        for record in answers:
            problem = all_problems[record.sample["task_id"]]
            entry_point = problem.get("entry_point")
            code = _extract_code(record.raw_text, entry_point=entry_point)
            completion = _normalize_completion(code)
            record.sample["completion"] = completion

        _write_jsonl(samples_path, (rec.sample for rec in answers))
        _write_jsonl(
            problems_path,
            (all_problems[rec.sample["task_id"]] for rec in answers),
        )

        details: List[Dict[str, Any]] = []
        per_task_results: Dict[str, List[bool]] = {}

        for record in answers:
            task_id = record.sample["task_id"]
            problem = all_problems[task_id]
            completion = record.sample["completion"]
            result = _run_completion(problem, completion, timeout=3.0)
            details.append(result)
            per_task_results.setdefault(task_id, []).append(bool(result["passed"]))

    total_tasks = len(per_task_results)
    pass_at_1 = 0.0
    if total_tasks:
        pass_at_1 = sum(1.0 if any(outcomes) else 0.0 for outcomes in per_task_results.values()) / float(total_tasks)

    metrics = {
        "bench_name": run_metadata["bench_name"],
        "model_id": run_metadata["model_id"],
        "variant": run_metadata["variant"],
        "answer_file": str(answer_file),
        "question_file": str(question_file),
        "num_questions": len(answers),
        "pass@1": pass_at_1,
        "details": details,
        "generation_stats": run_metadata.get("generation_stats", {}),
        "config": run_metadata.get("config", {}),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics
