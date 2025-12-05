"""Unified evaluation script for EAGLE-3 models across multiple benchmarks.

Supports: mt_bench, gsm8k, humaneval, alpaca, qa, sum

Usage:
python eval_eagle.py \
    --base-model-path allenai/OLMoE-1B-7B-0125-Instruct \
    --ea-model-path wantsleep/OLMoE_1B_7B_Eagle3 \
    --model-id olmoe-1b-eagle3 \
    --bench-name mt_bench \
    --use-eagle
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from tqdm import tqdm

# Ensure repository root is available on sys.path for package imports
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parents[1]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from fastchat.llm_judge.common import load_questions  # type: ignore

from fastchat.model import get_conversation_template  # type: ignore

import shortuuid  # type: ignore

from eagle.model.ea_model import EaModel
from eagle.model.utils import prepare_logits_processor
from eagle.benchmark.scorers import get_scorer


# Model name mappings for automatic model ID generation
MODEL_NAME_MAP = {
    "allenai/OLMoE-1B-7B-0125-Instruct": "olmoe1b7b",
    "Qwen/Qwen3-30B-A3B": "qwen3-30b-a3b",
}

# Default max_new_tokens per benchmark
BENCHMARK_MAX_TOKENS = {
    "mt_bench": 1024,
    "gsm8k": 1024,
    "humaneval": 512,
    "alpaca": 1024,
    "qa": 1024,
    "sum": 1024,
}


def get_model_id(
    base_model_path: str,
    use_eagle: bool,
    total_token: int,
    expert_selection_mode: str,
    expert_count_budget: Optional[int],
    expert_probability_budget: Optional[float],
    expert_pruning_strategy: str,
    temperature: float = 1.0,
) -> str:
    """Generate model ID from base model path and configuration."""
    if base_model_path not in MODEL_NAME_MAP:
        raise ValueError(
            f"Unknown model: {base_model_path}\n"
            f"Supported models: {list(MODEL_NAME_MAP.keys())}"
        )

    base_id = MODEL_NAME_MAP[base_model_path]

    # Build the model ID components
    parts = [base_id]

    # Add temperature first (always explicit)
    # Format temperature cleanly (e.g., 0.0 -> t0p0, 0.8 -> t0p8, 1.0 -> t1p0)
    temp_str = f"t{temperature:.1f}".replace(".", "p")
    parts.append(temp_str)

    # Add mode (eagle3 or autoregressive)
    mode = "eagle3" if use_eagle else "autoregressive"
    parts.append(mode)

    # Only add EAGLE-specific parameters for EAGLE mode
    if use_eagle:
        # Add number of draft tokens
        parts.append(f"draft{total_token}")

        # Always add expert selection parameters (be explicit, no hiding defaults)
        if expert_count_budget is not None:
            parts.append(f"cap{expert_count_budget}")
        elif expert_probability_budget is not None:
            # Format as percentage without decimal (e.g., 0.95 -> p95)
            pct = int(expert_probability_budget * 100)
            parts.append(f"p{pct}")
        else:
            # No expert pruning
            parts.append("nocap")

        # Always add pruning strategy
        parts.append(expert_pruning_strategy)

    return "-".join(parts)


@torch.inference_mode()
def run_evaluation(args):
    """Run MT-Bench evaluation with optional EAGLE acceleration."""

    # Load model
    print(f"Loading model from {args.base_model_path}")
    print(f"EAGLE adapter: {args.ea_model_path if args.use_eagle else 'None (baseline)'}")

    # Load via EaModel wrapper (provides optimized KV-cache implementation)
    # Baseline mode uses naivegenerate(), EAGLE mode uses eagenerate()

    # Baseline requires an EAGLE checkpoint path to initialize the wrapper
    if not args.use_eagle:
        from transformers import AutoConfig
        Type = AutoConfig.from_pretrained(args.base_model_path).architectures[0]

        if Type == 'OlmoeForCausalLM' and not args.ea_model_path:
            args.ea_model_path = 'wantsleep/OLMoE_1B_7B_Eagle3'
            print(f"Using {args.ea_model_path} for wrapper initialization")

    if not args.ea_model_path:
        raise ValueError("--ea-model-path is required to initialize the wrapper.")

    model = EaModel.from_pretrained(
        use_eagle3=True,
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=args.total_token,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()

    # Configure expert selection/pruning
    model.set_expert_pruning_strategy(args.expert_pruning_strategy)
    if args.expert_selection_mode == "probability":
        model.set_probability_expert_selection(target=args.expert_probability_budget)
        print(f"Applying probability-based expert selection with target={args.expert_probability_budget}")
    elif args.expert_selection_mode == "cardinality" and args.expert_count_budget is not None:
        model.set_expert_count_budget(args.expert_count_budget)
        print(f"Applying cardinality-based expert selection with budget={args.expert_count_budget}")
    print(f"Using expert pruning strategy: {args.expert_pruning_strategy}")
    if args.oracle_trace_file:
        model.configure_oracle(
            trace_file=args.oracle_trace_file,
            choice_index=args.oracle_choice_index,
            strict=args.oracle_strict,
        )
        trace_count = len(model.oracle.trace_map) if model.oracle is not None else 0
        print(f"Oracle trace replay enabled ({trace_count} turn traces loaded from {args.oracle_trace_file}).")

    run_path = Path(args.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    answer_file = run_path / "answers.jsonl"
    summary_file = run_path / "summary.json"
    metrics_file = run_path / "metrics.json"
    config_file = run_path / "config.json"
    if answer_file.exists():
        answer_file.unlink()

    config_payload = {
        "model_id": args.model_id,
        "bench_name": args.bench_name,
        "base_model_path": args.base_model_path,
        "ea_model_path": args.ea_model_path,
        "conv_template": args.conv_template,
        "question_file": args.question_file,
        "num_questions": args.num_questions,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "use_eagle": args.use_eagle,
        "total_token": args.total_token,
        "expert_selection_mode": args.expert_selection_mode,
        "expert_count_budget": args.expert_count_budget,
        "expert_probability_budget": args.expert_probability_budget,
        "expert_pruning_strategy": args.expert_pruning_strategy,
    }
    config_file.write_text(json.dumps(config_payload, indent=2))

    tokenizer = model.get_tokenizer()

    # Detect if this is a Llama-3, OLMoE, or Qwen3 model (all use tokenizer chat templates)
    is_llama3 = "llama-3" in args.base_model_path.lower() or "llama3" in args.base_model_path.lower()
    is_olmoe = "olmoe" in args.base_model_path.lower()
    is_qwen3 = "qwen3" in args.base_model_path.lower()
    use_chat_template = is_llama3 or is_olmoe or is_qwen3

    if use_chat_template:
        print(f"Detected {args.base_model_path} - using tokenizer chat template")

    # Load questions
    questions = load_questions(args.question_file, args.question_begin, args.question_end)
    if args.num_questions:
        questions = questions[:args.num_questions]

    print(f"Evaluating {len(questions)} questions with temperature={args.temperature}")
    print(f"Conversation template: {args.conv_template}")
    print(f"Max new tokens: {args.max_new_tokens}")

    training_rows: List[Dict[str, Any]] = []
    training_parquet_path: Optional[str] = None
    if args.collect_expert_traces and args.trace_schema == "training":
        base_name, _ = os.path.splitext(answer_file)
        training_parquet_path = base_name + ".training.parquet"
        Path(training_parquet_path).unlink(missing_ok=True)

    # Track statistics
    all_stats = {
        'total_tokens': [],
        'total_iterations': [],
        'wall_time': [],
        'tokens_per_second': [],
        'active_experts': [],
    }

    if args.use_eagle:
        all_stats['accept_lengths'] = []
        all_stats['avg_accept_length'] = []
        all_stats['tokens_per_iter'] = []
        all_stats['speedup'] = []

    # Evaluate questions

    for record_index, question in enumerate(tqdm(questions, desc="Evaluating")):
        torch.manual_seed(0)
        if use_chat_template:
            messages = []
        else:
            conv = get_conversation_template(args.conv_template)

        turns = []
        turns_stats = []

        for turn_idx, turn_text in enumerate(question["turns"]):
            if use_chat_template:
                messages.append({"role": "user", "content": turn_text})
                # For Qwen3, disable thinking mode for faster, more direct responses
                enable_thinking = False if is_qwen3 else None
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
            else:
                conv.append_message(conv.roles[0], turn_text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

            torch.cuda.synchronize()
            start_time = time.time()

            oracle_turn_active = False
            if args.use_eagle and model.has_oracle():
                oracle_turn_active = model.start_oracle_turn(question["question_id"], turn_idx)

            if args.use_eagle:
                output_ids, new_tokens, iterations, accept_lengths, iteration_traces = model.eagenerate(
                    torch.as_tensor(input_ids).to(model.base_model.device),
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    log=True,
                    is_llama3=is_llama3,
                    collect_expert_traces=args.collect_expert_traces,
                    trace_schema=args.trace_schema,
                )
            else:
                # Baseline: use naivegenerate (optimized AR decoding)
                output_ids, new_tokens, iterations = model.naivegenerate(
                    torch.as_tensor(input_ids).to(model.base_model.device),
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    log=True,
                    is_llama3=is_llama3
                )
                accept_lengths = None

            oracle_stats = None
            if args.use_eagle and model.has_oracle():
                oracle_stats = model.finish_oracle_turn()

            # Decode output
            output_ids = output_ids[0][len(input_ids[0]):]

            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            # Handle stop tokens (for models not using chat templates)
            if not use_chat_template and conv.stop_token_ids:
                stop_indices = [i for i, id in enumerate(output_ids) if id in conv.stop_token_ids]
                if stop_indices:
                    output_ids = output_ids[:stop_indices[0]]

            output = tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)

            # Clean up output (for models not using chat templates)
            if not use_chat_template and conv.stop_str and conv.stop_str in output:
                output = output[:output.find(conv.stop_str)]

            output = output.strip()
            turns.append(output)

            # Track statistics
            # Ensure all values are Python types, not tensors
            new_tokens_val = new_tokens.item() if isinstance(new_tokens, torch.Tensor) else int(new_tokens)
            iterations_val = iterations.item() if isinstance(iterations, torch.Tensor) else int(iterations)

            throughput = new_tokens_val / elapsed if elapsed > 0 else 0
            turn_stats = {
                'tokens': new_tokens_val,
                'iterations': iterations_val,
                'time': elapsed,
                'throughput': throughput,
            }
            turn_stats['expert_selection_mode'] = args.expert_selection_mode
            turn_stats['expert_pruning_strategy'] = args.expert_pruning_strategy

            if args.use_eagle and oracle_stats and oracle_stats.get("expected_iterations", 0) > 0:
                oracle_payload = dict(oracle_stats)
                oracle_payload["turn_active"] = bool(oracle_turn_active)
                turn_stats['oracle'] = oracle_payload

            if args.use_eagle and accept_lengths is not None:
                # Convert list of tensors to CPU if needed
                if isinstance(accept_lengths, list):
                    accept_lengths = [x.item() if isinstance(x, torch.Tensor) else x for x in accept_lengths]
                elif isinstance(accept_lengths, torch.Tensor):
                    accept_lengths = accept_lengths.cpu().tolist()

                avg_accept = np.mean(accept_lengths)
                # Tokens per iteration = accepted tokens + 1 (the base token)
                tokens_per_iter = avg_accept + 1
                # Speedup vs baseline
                speedup = new_tokens_val / iterations_val if iterations_val > 0 else 0

                turn_stats['avg_accept_length'] = float(avg_accept)
                turn_stats['tokens_per_iter'] = float(tokens_per_iter)
                turn_stats['speedup'] = float(speedup)
                turn_stats['accept_lengths'] = [int(x) for x in accept_lengths]
                if args.collect_expert_traces:
                    turn_stats['trace_schema'] = args.trace_schema
                    if args.trace_schema == "training" and training_parquet_path is not None:
                        if iteration_traces is not None:
                            for trace in iteration_traces:
                                if isinstance(trace, dict) and trace.get("schema") == "training":
                                    iteration_idx = trace["iteration"]
                                    node_features = trace["node_features"]
                                    for node_entry in node_features:
                                        row = {
                                            "dataset": args.bench_name,
                                            "trace_file": training_parquet_path,
                                            "record_index": record_index,
                                            "choice_index": turn_idx,
                                            "stats_index": turn_idx,
                                            "question_id": question["question_id"],
                                            "model_id": args.model_id,
                                            "iteration": iteration_idx,
                                            "wave_index": iteration_idx,
                                            "trace_schema": "training",
                                        }
                                        row.update(node_entry)
                                        training_rows.append(row)
                        turn_stats['trace_pointer'] = training_parquet_path
                        turn_stats['expert_traces'] = []
                    else:
                        turn_stats['expert_traces'] = iteration_traces
            if args.use_eagle:
                cap_usage = model.pop_last_cap_usage_means()
                if cap_usage:
                    turn_stats['avg_active_experts'] = float(np.mean(cap_usage))
                if turn_stats.get('avg_active_experts') is not None:
                    all_stats['active_experts'].append(float(turn_stats['avg_active_experts']))

            turns_stats.append(turn_stats)

            # Update conversation
            if use_chat_template:
                messages.append({"role": "assistant", "content": output})
            else:
                conv.messages[-1][-1] = output

        # Save answer
        ans_json = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": args.model_id,
            "choices": [{
                "index": 0,
                "turns": turns,
                "stats": turns_stats
            }],
            "tstamp": time.time(),
        }

        with answer_file.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(ans_json) + "\n")

        for turn_stats in turns_stats:
            all_stats['total_tokens'].append(turn_stats['tokens'])
            all_stats['total_iterations'].append(turn_stats['iterations'])
            all_stats['wall_time'].append(turn_stats['time'])
            all_stats['tokens_per_second'].append(turn_stats['throughput'])

            if args.use_eagle:
                all_stats['avg_accept_length'].append(turn_stats['avg_accept_length'])
                all_stats['tokens_per_iter'].append(turn_stats['tokens_per_iter'])
                all_stats['speedup'].append(turn_stats['speedup'])

    if training_rows and training_parquet_path is not None:
        df = pd.DataFrame(training_rows)
        df.to_parquet(training_parquet_path, index=False)
        print(f"Wrote training traces to {training_parquet_path} ({len(df)} rows)")

    # Compute summary statistics
    mean_throughput_val = float(np.mean(all_stats['tokens_per_second'])) if all_stats['tokens_per_second'] else 0.0
    mean_accept_val = float(np.mean(all_stats['tokens_per_iter'])) if args.use_eagle and all_stats.get('tokens_per_iter') else 0.0

    total_tokens = sum(all_stats['total_tokens'])
    total_iterations = sum(all_stats['total_iterations'])
    total_time = sum(all_stats['wall_time'])
    assert len(all_stats['tokens_per_second']) > 0, "No timing stats collected"
    mean_throughput = float(np.mean(all_stats['tokens_per_second']))
    median_throughput = float(np.median(all_stats['tokens_per_second']))

    if args.use_eagle:
        mean_accept = float(np.mean(all_stats['avg_accept_length']))
        median_accept = float(np.median(all_stats['avg_accept_length']))
        mean_tokens_per_iter = float(np.mean(all_stats['tokens_per_iter']))
    else:
        mean_accept = 0.0
        median_accept = 0.0
        mean_tokens_per_iter = 0.0

    mean_active_experts = float(np.mean(all_stats['active_experts'])) if len(all_stats['active_experts']) > 0 else None

    summary_payload = {
        "model_id": args.model_id,
        "bench_name": args.bench_name,
        "questions_evaluated": len(questions),
        "turns": len(all_stats['total_tokens']),
        "mean_throughput": mean_throughput,
        "median_throughput": median_throughput,
        "mean_accept_length": mean_accept,
        "median_accept_length": median_accept,
        "mean_tokens_per_iter": mean_tokens_per_iter,
        "mean_active_experts": mean_active_experts,
        "total_tokens": total_tokens,
        "total_iterations": total_iterations,
        "total_time": total_time,
        "expert_selection_mode": args.expert_selection_mode,
        "expert_count_budget": args.expert_count_budget,
        "expert_probability_budget": args.expert_probability_budget,
        "expert_pruning_strategy": args.expert_pruning_strategy,
        "use_eagle": args.use_eagle,
        "artifacts": {
            "answers": str(answer_file),
            "summary": str(summary_file),
            "metrics": str(metrics_file),
            "config": str(config_file),
        },
    }
    summary_file.write_text(json.dumps(summary_payload, indent=2))

    run_metadata = {
        "bench_name": args.bench_name,
        "model_id": args.model_id,
        "variant": "standard",  # Default variant identifier
        "generation_stats": summary_payload,
        "config": config_payload,
    }
    scorer = get_scorer(args.bench_name)
    if scorer is not None:
        metrics = scorer(
            answer_file=answer_file,
            question_file=Path(args.question_file),
            output_path=metrics_file,
            run_metadata=run_metadata,
        )
    else:
        metrics = {
            "message": f"No scorer registered for dataset '{args.bench_name}'",
            "answer_file": str(answer_file),
        }
        metrics_file.write_text(json.dumps(metrics, indent=2))

    # Print consolidated summary
    print(f"\n{'='*70}")
    print(f"Results: {args.model_id}")
    print(f"{'='*70}")
    print(f"Questions: {len(questions)} | Tokens: {total_tokens} | Time: {total_time:.1f}s")
    print(f"Throughput: {mean_throughput:.1f} tok/s", end="")
    if args.use_eagle and mean_accept_val > 0:
        print(f" | Accept: {mean_accept_val:.2f}", end="")

    # Print key metric
    if "pass@1" in metrics:
        print(f" | Pass@1: {metrics['pass@1']:.3f}")
    elif "accuracy" in metrics:
        print(f" | Accuracy: {metrics['accuracy']:.3f}")
    elif "f1" in metrics:
        print(f" | F1: {metrics['f1']:.3f}")
    elif "rouge_l" in metrics:
        rouge_val = metrics["rouge_l"].get("f1", 0) if isinstance(metrics["rouge_l"], dict) else metrics["rouge_l"]
        print(f" | ROUGE-L: {rouge_val:.3f}")
    else:
        print()

    print(f"Saved to: {run_path}")
    print(f"{'='*70}")

    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation script for EAGLE models across multiple benchmarks")

    # Model arguments
    parser.add_argument("--base-model-path", type=str, required=True,
                        help="Path or HF repo ID for base model")
    parser.add_argument("--ea-model-path", type=str, default=None,
                        help="Path or HF repo ID for EAGLE adapter (optional for baseline)")
    parser.add_argument("--conv-template", type=str, default="vicuna",
                        help="Conversation template (vicuna, llama-2, llama-3, etc.)")

    # Evaluation arguments
    parser.add_argument("--bench-name", type=str, default="mt_bench",
                        help="Benchmark name")
    parser.add_argument("--question-begin", type=int, default=None,
                        help="A debug option. The begin index of questions.")
    parser.add_argument("--question-end", type=int, default=None,
                        help="A debug option. The end index of questions.")
    parser.add_argument("--num-questions", type=int, default=None,
                        help="Number of questions to evaluate (default: all, ignored if question-begin/end used)")

    # Generation arguments
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")

    # EAGLE arguments
    parser.add_argument("--use-eagle", action="store_true",
                        help="Use EAGLE-3 acceleration (requires --ea-model-path)")
    parser.add_argument("--total-token", type=int, default=63,
                        help="Number of draft tokens for EAGLE")
    parser.add_argument("--collect-expert-traces", action="store_true",
                        help="Capture per-iteration draft tree and routing traces (adds overhead)")
    parser.add_argument("--trace-schema", choices=["analysis", "training"], default="analysis",
                        help="Trace export format when collecting expert traces (analysis = full tree, training = lean node features)")
    parser.add_argument("--expert-selection-mode", choices=["cardinality", "probability"], default="cardinality",
                        help="Expert selection mode: cardinality (fixed count) or probability (coverage-based)")
    parser.add_argument("--expert-count-budget", type=int, default=None,
                        help="Number of experts to keep when using cardinality selection mode")
    parser.add_argument("--expert-probability-budget", type=float, default=None,
                        help="Fraction of routing probability to preserve when using probability selection mode (0-1]")
    parser.add_argument("--expert-pruning-strategy", choices=["substitution", "truncation"], default="substitution",
                        help="Pruning strategy: substitution (rerank within shortlist) or truncation (allow sparse activation)")
    parser.add_argument("--oracle-trace-file", type=str, default=None,
                        help="Path to a JSONL answer log containing expert_traces for oracle replay pruning")
    parser.add_argument("--oracle-choice-index", type=int, default=0,
                        help="Choice index to read from when loading oracle traces")
    parser.add_argument("--oracle-strict", action="store_true",
                        help="Fail if oracle trace for a turn is missing or exhausted during replay")

    args = parser.parse_args()

    # Set benchmark-specific max_new_tokens
    args.max_new_tokens = BENCHMARK_MAX_TOKENS.get(args.bench_name, 1024)

    if args.trace_schema == "training" and not args.collect_expert_traces:
        parser.error("--trace-schema=training requires --collect-expert-traces")

    # Validate expert selection/pruning arguments
    if args.expert_selection_mode == "cardinality":
        if args.expert_probability_budget is not None:
            parser.error("--expert-probability-budget requires --expert-selection-mode probability")
        # For cardinality mode, budget is optional (None means no pruning)
    elif args.expert_selection_mode == "probability":
        if args.expert_probability_budget is None:
            parser.error("--expert-probability-budget is required when --expert-selection-mode probability")
        if not (0 < args.expert_probability_budget <= 1):
            parser.error("--expert-probability-budget must be in range (0, 1]")
        if args.expert_count_budget is not None:
            parser.error("--expert-count-budget cannot be combined with --expert-selection-mode probability")

    # Validate arguments
    if args.use_eagle and not args.ea_model_path:
        parser.error("--use-eagle requires --ea-model-path")

    # Generate model ID
    args.model_id = get_model_id(
        base_model_path=args.base_model_path,
        use_eagle=args.use_eagle,
        total_token=args.total_token,
        expert_selection_mode=args.expert_selection_mode,
        expert_count_budget=args.expert_count_budget,
        expert_probability_budget=args.expert_probability_budget,
        expert_pruning_strategy=args.expert_pruning_strategy,
        temperature=args.temperature,
    )

    # Setup paths
    args.question_file = os.path.join(str(script_dir.parent), f"data/{args.bench_name}/question.jsonl")
    args.run_dir = os.path.join("results", args.bench_name, args.model_id)
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("="*70)
    print("CONFIGURATION:")
    print("="*70)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("="*70)

    # Run evaluation
    run_evaluation(args)
