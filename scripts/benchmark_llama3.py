#!/usr/bin/env python
"""
Benchmark baseline vs. EAGLE decoding for Llama 3.1 8B Instruct.

Example:
    python scripts/benchmark_llama3.py
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eagle.model.ea_model import EaModel

try:
    from fastchat.model import get_conversation_template
except ImportError as exc:
    raise RuntimeError(
        "fastchat is required for conversation templating. "
        "Install it via `pip install fastchat`."
    ) from exc


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Prompt file {prompt_path} is empty.")
        return prompt
    return args.prompt


def _format_prompt(prompt: str, template: str) -> str:
    conv = get_conversation_template(template)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def _prepare_input_ids(model: EaModel, prompt: str) -> torch.Tensor:
    encoded = model.tokenizer([prompt], return_tensors="pt")
    return encoded.input_ids.to(model.base_model.device)


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_generation(
    run_fn,
    warmup_input_ids: torch.Tensor,
    timed_input_ids: torch.Tensor,
    warmup_kwargs: Dict[str, Any],
    timed_kwargs: Dict[str, Any],
) -> Tuple[Any, float]:
    _ = run_fn(warmup_input_ids, **warmup_kwargs)

    _synchronize()
    start = time.perf_counter()
    outputs = run_fn(timed_input_ids, **timed_kwargs)
    _synchronize()
    return outputs, time.perf_counter() - start


def benchmark_mode(
    model: EaModel,
    mode: str,
    input_ids: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    timed_input = input_ids.clone()
    warmup_input = input_ids.clone()

    if mode == "baseline":
        run_kwargs = dict(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            log=True,
        )
        warmup_kwargs = dict(run_kwargs)
        warmup_kwargs["log"] = False

        outputs, elapsed = _time_generation(
            model.naivegenerate,
            warmup_input_ids=warmup_input,
            timed_input_ids=timed_input,
            warmup_kwargs=warmup_kwargs,
            timed_kwargs=run_kwargs,
        )
        generated_ids, tokens_generated, iterations = outputs
        return {
            "mode": "baseline",
            "tokens_generated": int(tokens_generated),
            "iterations": int(iterations),
            "tokens_per_sec": tokens_generated / elapsed if elapsed > 0 else float("nan"),
            "elapsed_sec": elapsed,
            "output_text": model.tokenizer.decode(generated_ids[0], skip_special_tokens=False),
        }

    if mode == "eagle":
        run_kwargs = dict(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            log=True,
            collect_expert_traces=False,
        )
        warmup_kwargs = dict(run_kwargs)
        warmup_kwargs["log"] = False

        outputs, elapsed = _time_generation(
            model.eagenerate,
            warmup_input_ids=warmup_input,
            timed_input_ids=timed_input,
            warmup_kwargs=warmup_kwargs,
            timed_kwargs=run_kwargs,
        )
        (
            generated_ids,
            tokens_generated,
            iterations,
            accept_lengths,
            _,
        ) = outputs
        acceptance_mean = (
            float(sum(accept_lengths) / len(accept_lengths)) if accept_lengths else 0.0
        )
        return {
            "mode": "eagle",
            "tokens_generated": int(tokens_generated),
            "iterations": int(iterations),
            "tokens_per_sec": tokens_generated / elapsed if elapsed > 0 else float("nan"),
            "elapsed_sec": elapsed,
            "avg_accept_length": acceptance_mean,
            "output_text": model.tokenizer.decode(generated_ids[0], skip_special_tokens=False),
        }

    raise ValueError(f"Unknown benchmark mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs. EAGLE decoding for Llama 3.1 8B Instruct.")
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face identifier or path for the base model.",
    )
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        help="Hugging Face identifier or path for the EAGLE adapter.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Why do mathematicians care about prime numbers?",
        help="Prompt string to benchmark with.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional path to a file containing the prompt. Overrides --prompt.",
    )
    parser.add_argument(
        "--conv-template",
        type=str,
        default="vicuna",
        help="Conversation template name to wrap the prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Nucleus sampling parameter.",
    )
    parser.add_argument(
        "--top-k",
        type=float,
        default=0.0,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map passed to `EaModel.from_pretrained`.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype for model weights (float16, bfloat16, float32).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["baseline", "eagle", "both"],
        help="Benchmark baseline, EAGLE, or both.",
    )
    parser.add_argument(
        "--dump-json",
        type=str,
        default=None,
        help="Optional path to save benchmark results as JSON.",
    )
    parser.add_argument(
        "--print-output",
        action="store_true",
        help="Print decoded text for each run.",
    )
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype {name}. Choose from {sorted(mapping)}.")
    return mapping[key]


def main() -> None:
    args = parse_args()
    prompt = _load_prompt(args)
    formatted_prompt = _format_prompt(prompt, args.conv_template)

    dtype = _resolve_dtype(args.dtype)
    model = EaModel.from_pretrained(
        use_eagle3=True,
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        total_token=63,
    )
    model.eval()

    input_ids = _prepare_input_ids(model, formatted_prompt)

    modes = ["baseline", "eagle"] if args.mode == "both" else [args.mode]
    results = []

    for mode in modes:
        metrics = benchmark_mode(model, mode, input_ids, args)
        summary = (
            f"{mode.upper():>8}: tokens={metrics['tokens_generated']} "
            f"elapsed={metrics['elapsed_sec']:.2f}s "
            f"throughput={metrics['tokens_per_sec']:.2f} tok/s"
        )
        if mode == "eagle" and "avg_accept_length" in metrics:
            summary += f" avg_accept_length={metrics['avg_accept_length']:.2f}"
        print(summary)

        if args.print_output:
            print(f"\n--- {mode.upper()} OUTPUT ---")
            print(metrics["output_text"])
            print("---------------------------\n")

        results.append(metrics)

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.dump_json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
