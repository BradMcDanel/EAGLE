#!/usr/bin/env python3
"""Stream conversational datasets into EAGLE-friendly JSONL shards.

This utility loads chat-style corpora from Hugging Face, selects user turns, and
emits question records compatible with
``eagle/evaluation/eval_eagle.py``. Each record contains a list of user prompts
(``turns``) that will be replayed sequentially during instrumentation.

Example:

    python scripts/stream_conversation_dataset.py \\
        --dataset ultrachat \\
        --sample-count 5000 \\
        --records-per-shard 1000 \\
        --output-dir results/ultrachat/prompts

By default the script samples entire conversations, but ``--window-size`` can be
used to draw contiguous slices of user turns (helpful when conversations are
long). Sharding parameters keep memory usage predictable when collecting large
datasets.

The loader fetches datasets into memory for stability; pass ``--use-streaming``
to opt into Hugging Face's streaming iterators instead (useful when the full
dataset cannot fit locally).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration describing how to parse a conversational dataset."""

    name: str
    path: str
    default_split: str
    message_field: str
    role_field: str
    text_fields: Sequence[str]
    user_roles: Sequence[str]
    assistant_roles: Sequence[str]
    id_field: str = "id"


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "ultrachat": DatasetConfig(
        name="ultrachat",
        path="HuggingFaceH4/ultrachat_200k",
        default_split="train_sft",
        message_field="messages",
        role_field="from",
        text_fields=("content", "value", "text"),
        user_roles=("human", "user"),
        assistant_roles=("gpt", "assistant"),
    ),
    "sharegpt": DatasetConfig(
        name="sharegpt",
        path="anon8231489123/ShareGPT_Vicuna_unfiltered",
        default_split="train",
        message_field="conversations",
        role_field="from",
        text_fields=("value", "content", "text"),
        user_roles=("human", "user"),
        assistant_roles=("gpt", "assistant", "assistant_legacy"),
    ),
}


MAX_CONTEXT_TOKENS = 2048
GENERATION_RESERVED_TOKENS = 256
PROMPT_TOKEN_LIMIT = MAX_CONTEXT_TOKENS - GENERATION_RESERVED_TOKENS
TOKENIZER_MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a conversational dataset and emit JSONL shards with user turns."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS.keys()),
        required=True,
        help="Dataset key to load (see DATASET_CONFIGS).",
    )
    parser.add_argument(
        "--split",
        help="Dataset split to load. Defaults to the config's recommended split.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Maximum number of conversation windows to export (0 = unlimited).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="Optional number of user turns per record. When 0, the full user "
        "conversation is emitted.",
    )
    parser.add_argument(
        "--min-user-turns",
        type=int,
        default=1,
        help="Discard conversations with fewer user turns than this value.",
    )
    parser.add_argument(
        "--records-per-shard",
        type=int,
        default=1000,
        help="Number of exported records per JSONL shard file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where shard files will be written.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=0,
        help="Optional dataset-level sharding factor (passed to datasets.shard).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Optional shard index to load when --num-shards is provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for window sampling.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help="When streaming, size of the shuffle buffer; otherwise toggles "
        "random ordering (0 = preserve dataset order).",
    )
    parser.add_argument(
        "--use-streaming",
        action="store_true",
        help="Enable HF streaming mode. When omitted, datasets are loaded in-memory "
        "for greater stability.",
    )
    return parser.parse_args()


def normalize_role(message: Dict[str, str], role_field: str) -> Optional[str]:
    raw = message.get(role_field)
    if raw is None and role_field != "role":
        raw = message.get("role")
    if raw is None and role_field != "speaker":
        raw = message.get("speaker")
    if raw is None:
        return None
    return str(raw).strip().lower()


def extract_text(message: Dict[str, str], text_fields: Sequence[str]) -> Optional[str]:
    for key in text_fields:
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def count_prompt_tokens(turns: Sequence[str], tokenizer: AutoTokenizer) -> int:
    """Return tokenized length for a list of user turns using the base chat template."""
    messages = [{"role": "user", "content": turn} for turn in turns]
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
    except AttributeError:
        prompt_text = "\n\n".join(turns)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

    if isinstance(prompt_ids, list):
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        return len(prompt_ids)

    try:
        return len(prompt_ids.tolist())
    except AttributeError:
        return int(prompt_ids.shape[0])  # type: ignore[union-attr]


def iter_conversation_turns(
    records: Iterable[Dict[str, object]],
    config: DatasetConfig,
    window_size: int,
    min_user_turns: int,
    rng: random.Random,
    tokenizer: Optional[AutoTokenizer] = None,
    max_prompt_tokens: Optional[int] = None,
    stats: Optional[Dict[str, int]] = None,
) -> Iterator[Dict[str, object]]:
    for index, record in enumerate(records):
        messages = record.get(config.message_field)
        if not isinstance(messages, list):
            continue

        user_turns: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = normalize_role(message, config.role_field)
            if role in config.user_roles:
                text = extract_text(message, config.text_fields)
                if text:
                    user_turns.append(text)

        if len(user_turns) < min_user_turns:
            continue

        sample_turns: List[str]
        if window_size and window_size > 0:
            if len(user_turns) < window_size:
                continue
            start = rng.randint(0, len(user_turns) - window_size)
            sample_turns = user_turns[start : start + window_size]
            window_suffix = f"w{window_size}-s{start}"
        else:
            sample_turns = user_turns
            window_suffix = "full"

        if tokenizer is not None and max_prompt_tokens:
            prompt_tokens = count_prompt_tokens(sample_turns, tokenizer)
            if prompt_tokens > max_prompt_tokens:
                if stats is not None:
                    stats["token_overflow_skipped"] = stats.get("token_overflow_skipped", 0) + 1
                continue

        record_id = record.get(config.id_field) or record.get("conversation_id")
        if record_id is None:
            record_id = f"{config.name}-{index}"

        question_id = f"{config.name}-{record_id}-{window_suffix}"
        yield {
            "question_id": str(question_id),
            "dataset": config.name,
            "source_id": str(record_id),
            "turns": sample_turns,
        }


def load_records(args: argparse.Namespace, config: DatasetConfig) -> Iterable[Dict[str, object]]:
    split = args.split or config.default_split
    if args.use_streaming:
        dataset = load_dataset(
            config.path,
            split=split,
            streaming=True,
        )
        if args.num_shards:
            dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index, contiguous=True)
        if args.shuffle_buffer:
            dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        return dataset

    dataset = load_dataset(
        config.path,
        split=split,
    )
    if args.num_shards:
        dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index, contiguous=True)

    indices = list(range(len(dataset)))
    if args.shuffle_buffer:
        rng = random.Random(args.seed)
        rng.shuffle(indices)

    def generator() -> Iterator[Dict[str, object]]:
        for idx in indices:
            yield dataset[int(idx)]

    return generator()


def write_shards(
    samples: Iterable[Dict[str, object]],
    output_dir: Path,
    records_per_shard: int,
    sample_limit: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    count_in_shard = 0
    total_written = 0
    shard_file = None

    def open_new_shard(idx: int):
        path = output_dir / f"shard-{idx:04d}.jsonl"
        return path.open("w", encoding="utf-8")

    try:
        for sample in samples:
            if sample_limit and total_written >= sample_limit:
                break

            if shard_file is None:
                shard_file = open_new_shard(shard_index)
                count_in_shard = 0

            shard_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count_in_shard += 1
            total_written += 1

            if records_per_shard and count_in_shard >= records_per_shard:
                shard_file.close()
                shard_file = None
                shard_index += 1

    finally:
        if shard_file is not None and not shard_file.closed:
            shard_file.close()

    return total_written


def main() -> None:
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, use_fast=True)
    except Exception as exc:  # pragma: no cover - defensive guard for missing weights
        raise RuntimeError(
            f"Failed to load tokenizer '{TOKENIZER_MODEL_ID}'. "
            "Ensure the checkpoint is available locally."
        ) from exc

    rng = random.Random(args.seed)
    dataset_stream = load_records(args, config)
    stats: Dict[str, int] = {"token_overflow_skipped": 0}
    samples = iter_conversation_turns(
        dataset_stream,
        config=config,
        window_size=args.window_size,
        min_user_turns=args.min_user_turns,
        rng=rng,
        tokenizer=tokenizer,
        max_prompt_tokens=PROMPT_TOKEN_LIMIT,
        stats=stats,
    )
    total_written = write_shards(
        samples,
        output_dir=args.output_dir,
        records_per_shard=args.records_per_shard,
        sample_limit=args.sample_count,
    )

    if args.sample_count and total_written < args.sample_count:
        expected = args.sample_count
        msg = (
            f"Requested {expected} samples but only wrote {total_written}. "
            "Consider lowering --min-user-turns or --window-size."
        )
        print(msg)
    else:
        print(f"Wrote {total_written} samples to {args.output_dir}")

    skipped = stats.get("token_overflow_skipped", 0)
    if skipped:
        print(f"Skipped {skipped} conversations that exceeded the {PROMPT_TOKEN_LIMIT}-token prompt limit.")


if __name__ == "__main__":
    main()
