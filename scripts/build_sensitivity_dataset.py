#!/usr/bin/env python3
"""Assemble a short-form multi-domain prompt set for MoE sensitivity analysis."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import requests
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


PromptBuilder = Callable[[Dict], Optional[str]]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: str
    split: str
    category: str
    prompt_builder: PromptBuilder
    config: Optional[str] = None
    default_samples: int = 128
    remote_rows: Optional["RemoteRowsConfig"] = None


@dataclass(frozen=True)
class RemoteRowsConfig:
    dataset: str
    config: str
    split: str
    batch_size: int = 100
    fetch_multiplier: float = 2.0  # Fetch extra rows to survive filtering
    timeout: float = 30.0


def build_arc_prompt(example: Dict) -> Optional[str]:
    choices = example.get("choices") or {}
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    if not labels or not texts:
        return None
    options = "\n".join(f"({label}) {text}" for label, text in zip(labels, texts))
    question = example.get("question")
    if not question:
        return None
    return (
        "Answer the multiple-choice science question.\n"
        f"Question: {question.strip()}\nChoices:\n{options}"
    )


def build_piqa_prompt(example: Dict) -> Optional[str]:
    goal = example.get("goal")
    sol1 = example.get("sol1") or example.get("solution1")
    sol2 = example.get("sol2") or example.get("solution2")
    if not goal or not sol1 or not sol2:
        return None
    return (
        "Choose the option that best completes the goal.\n"
        f"Goal: {goal.strip()}\nChoices:\n(A) {sol1.strip()}\n(B) {sol2.strip()}"
    )


def build_hellaswag_prompt(example: Dict) -> Optional[str]:
    ctx = example.get("ctx", "")
    ctx_a = example.get("ctx_a", "")
    context = f"{ctx.strip()} {ctx_a.strip()}".strip()
    endings = example.get("endings") or []
    if not context or not endings:
        return None
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(f"({letters[i]}) {text}" for i, text in enumerate(endings))
    return f"Choose the best ending for the story.\nContext: {context}\nEndings:\n{options}"


def build_winogrande_prompt(example: Dict) -> Optional[str]:
    sentence = example.get("sentence")
    option1 = example.get("option1")
    option2 = example.get("option2")
    if not sentence or not option1 or not option2:
        return None
    sentence = sentence.replace("_", "_____")
    return (
        "Fill in the blank to make the sentence true.\n"
        f"Sentence: {sentence}\nChoices:\n(A) {option1}\n(B) {option2}"
    )


def build_commonsenseqa_prompt(example: Dict) -> Optional[str]:
    question = example.get("question")
    choices = example.get("choices") or {}
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    if not question or not labels or not texts:
        return None
    options = "\n".join(f"({label}) {text}" for label, text in zip(labels, texts))
    return f"Answer the commonsense question.\nQuestion: {question.strip()}\nChoices:\n{options}"


def build_boolq_prompt(example: Dict) -> Optional[str]:
    question = example.get("question")
    passage = example.get("passage")
    if not question:
        return None
    if passage:
        return (
            "Answer the question with yes or no.\n"
            f"Passage: {passage.strip()}\nQuestion: {question.strip()}"
        )
    return f"Answer the question with yes or no.\nQuestion: {question.strip()}"


def build_mbpp_prompt(example: Dict) -> Optional[str]:
    text = example.get("text")
    code = example.get("code", "")
    if not text:
        return None
    signature = ""
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("def "):
            signature = line
            break
    suffix = f"\nYou may find the target signature helpful: {signature}" if signature else ""
    return f"Write Python code for the following problem:\n{text.strip()}{suffix}"


def build_math_prompt(example: Dict) -> Optional[str]:
    problem = example.get("problem")
    if not problem:
        return None
    return f"Solve the math problem. Show your reasoning before the final answer.\nProblem: {problem.strip()}"


def build_xsum_prompt(example: Dict) -> Optional[str]:
    document = example.get("document")
    if not document:
        return None
    return f"Summarize the following news article in one concise paragraph:\n{document.strip()}"


def build_dolly_prompt(example: Dict) -> Optional[str]:
    instruction = example.get("instruction")
    if not instruction:
        return None
    parts = [f"Instruction: {instruction.strip()}"]
    context = example.get("context")
    if context:
        parts.append(f"Context: {context.strip()}")
    inp = example.get("input")
    if inp:
        parts.append(f"Input: {inp.strip()}")
    parts.append("Response:")
    return "\n\n".join(parts)


def build_openbookqa_prompt(example: Dict) -> Optional[str]:
    question = example.get("question_stem")
    choices = example.get("choices") or {}
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    if not question or not labels or not texts:
        return None
    fact = example.get("fact1")
    options = "\n".join(f"({label}) {text}" for label, text in zip(labels, texts))
    header = "Use the supplied science fact when helpful.\n" if fact else ""
    fact_line = f"Fact: {fact.strip()}\n" if fact else ""
    return (
        f"{header}{fact_line}"
        f"Question: {question.strip()}\nChoices:\n{options}\nAnswer with the best option."
    )


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        name="arc_challenge",
        hf_id="allenai/ai2_arc",
        config="ARC-Challenge",
        split="validation",
        category="science_mc",
        prompt_builder=build_arc_prompt,
    ),
    DatasetSpec(
        name="hellaswag",
        hf_id="Rowan/hellaswag",
        split="validation",
        category="commonsense",
        prompt_builder=build_hellaswag_prompt,
    ),
    DatasetSpec(
        name="winogrande",
        hf_id="allenai/winogrande",
        config="winogrande_xl",
        split="validation",
        category="coreference",
        prompt_builder=build_winogrande_prompt,
    ),
    DatasetSpec(
        name="commonsenseqa",
        hf_id="tau/commonsense_qa",
        split="validation",
        category="commonsense",
        prompt_builder=build_commonsenseqa_prompt,
    ),
    DatasetSpec(
        name="boolq",
        hf_id="google/boolq",
        split="validation",
        category="qa",
        prompt_builder=build_boolq_prompt,
    ),
    DatasetSpec(
        name="hendrycks_math_algebra",
        hf_id="EleutherAI/hendrycks_math",
        config="algebra",
        split="test",
        category="math",
        prompt_builder=build_math_prompt,
    ),
    DatasetSpec(
        name="xsum",
        hf_id="EdinburghNLP/xsum",
        config="default",
        split="train",
        category="summarization",
        prompt_builder=build_xsum_prompt,
        remote_rows=RemoteRowsConfig(
            dataset="EdinburghNLP/xsum",
            config="default",
            split="train",
        ),
    ),
    DatasetSpec(
        name="dolly15k",
        hf_id="databricks/databricks-dolly-15k",
        split="train",
        category="chat",
        prompt_builder=build_dolly_prompt,
    ),
    DatasetSpec(
        name="openbookqa",
        hf_id="allenai/openbookqa",
        config="main",
        split="validation",
        category="science_mc",
        prompt_builder=build_openbookqa_prompt,
    ),
]


def parse_limit_overrides(entries: Iterable[str]) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --dataset-limit entry: '{entry}' (use name=count)")
        name, value = entry.split("=", 1)
        overrides[name.strip()] = int(value)
    return overrides


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def resolve_tokenizer_path(tokenizer_arg: str) -> str:
    """Allow pointing at either an HF hub id or a local cache directory."""
    candidate = Path(tokenizer_arg)
    if not candidate.exists():
        return tokenizer_arg
    if (candidate / "tokenizer.json").exists() or (candidate / "tokenizer.model").exists():
        return str(candidate)
    snapshots = candidate / "snapshots"
    if snapshots.is_dir():
        subdirs = sorted(p for p in snapshots.iterdir() if p.is_dir())
        if subdirs:
            logging.info("Using tokenizer snapshot %s", subdirs[-1])
            return str(subdirs[-1])
    raise FileNotFoundError(
        f"Could not locate tokenizer files under {tokenizer_arg}. "
        "Pass the snapshot directory containing tokenizer.json."
    )


def fetch_remote_rows_dataset(
    config: RemoteRowsConfig,
    target: int,
    seed: int,
) -> Dataset:
    """Fetch rows from the HF datasets-server REST API and build a Dataset."""
    base_params = {
        "dataset": config.dataset,
        "config": config.config,
        "split": config.split,
    }
    meta_resp = requests.get(
        "https://datasets-server.huggingface.co/rows",
        params={**base_params, "offset": 0, "length": 0},
        timeout=config.timeout,
    )
    meta_resp.raise_for_status()
    meta = meta_resp.json()
    total_rows = int(meta.get("num_rows_total", 0) or 0)
    min_rows = max(int(target * config.fetch_multiplier), target + 32)
    if total_rows > min_rows:
        max_offset = max(total_rows - min_rows, 1)
        start_offset = (seed * 997) % max_offset
    else:
        start_offset = 0

    rows: List[Dict] = []
    offset = start_offset
    while len(rows) < min_rows:
        length = min(config.batch_size, min_rows - len(rows))
        resp = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={**base_params, "offset": offset, "length": length},
            timeout=config.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("rows", [])
        if not batch:
            break
        for entry in batch:
            record = dict(entry.get("row", {}))
            record["__source_row_index"] = entry.get("row_idx", offset)
            rows.append(record)
        offset += len(batch)
        if total_rows and offset >= total_rows:
            break

    if not rows:
        raise RuntimeError(
            f"Remote dataset {config.dataset} returned no rows (offset={start_offset})."
        )
    return Dataset.from_list(rows)


def load_dataset_for_spec(
    spec: DatasetSpec,
    target: int,
    seed: int,
) -> Optional[Dataset]:
    if spec.remote_rows:
        logging.info(
            "Fetching remote rows for %s via datasets-server (dataset=%s, split=%s)",
            spec.name,
            spec.remote_rows.dataset,
            spec.remote_rows.split,
        )
        try:
            dataset = fetch_remote_rows_dataset(spec.remote_rows, target, seed)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to fetch remote dataset %s: %s", spec.name, exc)
            return None
        return dataset.shuffle(seed=seed)

    logging.info("Loading %s (config=%s, split=%s)", spec.hf_id, spec.config, spec.split)
    try:
        dataset: Dataset = load_dataset(
            path=spec.hf_id,
            name=spec.config,
            split=spec.split,
        )  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to load %s (%s): %s", spec.hf_id, spec.split, exc)
        return None
    return dataset.shuffle(seed=seed)


def sample_dataset(
    spec: DatasetSpec,
    tokenizer,
    max_tokens: int,
    target: int,
    seed: int,
) -> List[Dict]:
    dataset = load_dataset_for_spec(spec, target, seed)
    if dataset is None:
        return []
    selected: List[Dict] = []
    for idx in range(len(dataset)):
        example = dataset[int(idx)]
        prompt = spec.prompt_builder(example)
        if not prompt:
            continue
        if token_count(tokenizer, prompt) > max_tokens:
            continue
        source_row_index = example.get("__source_row_index", idx)
        selected.append(
            {
                "prompt": prompt,
                "row_index": source_row_index,
            }
        )
        if len(selected) >= target:
            break
    if len(selected) < target:
        logging.warning(
            "Dataset %s provided %d / %d requested prompts (likely filtered by length).",
            spec.name,
            len(selected),
            target,
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a JSONL question file for MoE sensitivity analysis."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eagle/data/sensitivity/question.jsonl"),
        help="Path to the JSONL file to write.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="Tokenizer checkpoint used to enforce the max token constraint.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Maximum number of tokens allowed per prompt.",
    )
    parser.add_argument(
        "--default-samples",
        type=int,
        default=128,
        help="Default number of prompts to keep per dataset unless overridden.",
    )
    parser.add_argument(
        "--dataset-limit",
        action="append",
        default=[],
        metavar="NAME=COUNT",
        help="Override the number of prompts for a specific dataset (may be passed multiple times).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for dataset shuffling.",
    )
    args = parser.parse_args()

    overrides = parse_limit_overrides(args.dataset_limit)
    tokenizer_path = resolve_tokenizer_path(args.tokenizer)
    logging.info("Loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict] = []
    question_id = 0

    for spec_index, spec in enumerate(DATASETS):
        target = overrides.get(spec.name, args.default_samples if args.default_samples > 0 else spec.default_samples)
        prompts = sample_dataset(
            spec=spec,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            target=target,
            seed=args.seed + spec_index,
        )
        for sample in prompts:
            record = {
                "question_id": question_id,
                "category": spec.category,
                "turns": [sample["prompt"]],
                "source": {
                    "dataset": spec.hf_id,
                    "config": spec.config,
                    "split": spec.split,
                    "row_index": sample["row_index"],
                    "name": spec.name,
                },
            }
            all_records.append(record)
            question_id += 1

    with args.output.open("w", encoding="utf-8") as fout:
        for record in all_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info("Wrote %d prompts to %s", len(all_records), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
