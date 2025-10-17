"""Unified evaluation script for EAGLE models across multiple benchmarks.

Supports: mt_bench, gsm8k, humaneval, alpaca, qa, sum

Usage:
python eval_eagle.py \
    --base-model-path allenai/OLMoE-1B-7B-0125-Instruct \
    --ea-model-path wantsleep/OLMoE_1B_7B_Eagle3 \
    --model-id olmoe-1b-eagle3 \
    --conv-template vicuna \
    --bench-name mt_bench \
    --num-questions 10 \
    --use-eagle3
"""
import argparse
import json
import os
import sys
import time
import numpy as np

import shortuuid
import torch
from tqdm import tqdm

# Add parent directory to path for imports
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template

try:
    from eagle.model.ea_model import EaModel
    from eagle.model.utils import prepare_logits_processor
except:
    from model.ea_model import EaModel
    from model.utils import prepare_logits_processor


def configure_max_active_experts(model: torch.nn.Module, max_active_experts: int) -> int:
    """Propagate a per-wave expert budget to every supported MoE block."""
    updated = 0
    for module in model.modules():
        setter = getattr(module, "set_max_active_experts", None)
        if callable(setter):
            setter(max_active_experts)
            updated += 1
    return updated


@torch.inference_mode()
def run_evaluation(
    base_model_path,
    ea_model_path,
    model_id,
    conv_template,
    question_file,
    answer_file,
    num_questions,
    max_new_tokens,
    temperature,
    use_eagle,
    args
):
    """Run MT-Bench evaluation with optional EAGLE acceleration."""

    # Load model
    print(f"Loading model from {base_model_path}")
    print(f"EAGLE adapter: {ea_model_path if use_eagle else 'None (baseline)'}")

    # Always load via EaModel wrapper (provides optimized KV-cache implementation)
    # Baseline mode uses naivegenerate(), EAGLE mode uses eagenerate()

    # For baseline, we need a dummy EAGLE checkpoint path to initialize the wrapper
    # We can use any EAGLE checkpoint since we only use the base model
    if not use_eagle:
        # Check if OLMoE - if so, use a default EAGLE checkpoint for initialization
        from transformers import AutoConfig
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'OlmoeForCausalLM' and not ea_model_path:
            # Use default EAGLE3 checkpoint for OLMoE (only to initialize wrapper, not used in baseline)
            ea_model_path = 'wantsleep/OLMoE_1B_7B_Eagle3'
            args.use_eagle3 = True  # Match the checkpoint architecture
            print(f"Note: Using {ea_model_path} for wrapper initialization (only base model will be used)")

    if not ea_model_path:
        raise ValueError("--ea-model-path is required. For baseline, the same checkpoint path is needed to initialize the wrapper.")

    model = EaModel.from_pretrained(
        use_eagle3=args.use_eagle3,
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token if use_eagle else 60,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    tokenizer = model.get_tokenizer()

    if args.max_active_experts is not None:
        updated_blocks = configure_max_active_experts(model.base_model, args.max_active_experts)
        if updated_blocks:
            print(
                f"Capping active experts per wave to {args.max_active_experts}"
                f" across {updated_blocks} MoE blocks"
            )
        else:
            print("Warning: max_active_experts set but no MoE blocks support this option")

    # Detect if this is a Llama-3, OLMoE, or Qwen3 model (all use tokenizer chat templates)
    is_llama3 = "llama-3" in base_model_path.lower() or "llama3" in base_model_path.lower()
    is_olmoe = "olmoe" in base_model_path.lower()
    is_qwen3 = "qwen3" in base_model_path.lower()
    use_chat_template = is_llama3 or is_olmoe or is_qwen3

    if is_llama3:
        print("Detected Llama-3 model - will use tokenizer chat template and special stop token handling")
    elif is_olmoe:
        print("Detected OLMoE model - will use tokenizer chat template")
    elif is_qwen3:
        print("Detected Qwen3 model - will use tokenizer chat template")

    # Load questions
    questions = load_questions(question_file, args.question_begin, args.question_end)
    if num_questions:
        questions = questions[:num_questions]

    print(f"Evaluating {len(questions)} questions with temperature={temperature}")
    print(f"Conversation template: {conv_template}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Warmup tokens: {args.warmup_tokens}")

    # Warmup
    print("Warming up (2 cycles)...")
    question = questions[0]
    for warmup_idx in range(2):
        print(f"  Warmup cycle {warmup_idx + 1}/2...")
        if use_chat_template:
            # For models with native chat templates (Llama-3, OLMoE), use tokenizer chat template
            messages = []
        else:
            conv = get_conversation_template(conv_template)

        for turn_idx, turn in enumerate(question["turns"]):
            print(f"    Turn {turn_idx + 1}/{len(question['turns'])}...", end=" ", flush=True)

            if use_chat_template:
                messages.append({"role": "user", "content": turn})
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
                conv.append_message(conv.roles[0], turn)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

            start = time.time()
            if use_eagle:
                output_ids, _, _, _, _ = model.eagenerate(
                    torch.as_tensor(input_ids).to(model.base_model.device),
                    temperature=temperature,
                    max_new_tokens=args.warmup_tokens,
                    log=True,
                    is_llama3=is_llama3,
                    collect_expert_traces=args.collect_expert_traces,
                )
            else:
                # Baseline: use naivegenerate (optimized AR decoding)
                output_ids, _, _ = model.naivegenerate(
                    torch.as_tensor(input_ids).to(model.base_model.device),
                    temperature=temperature,
                    max_new_tokens=args.warmup_tokens,
                    log=True,
                    is_llama3=is_llama3
                )

            output = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
            elapsed = time.time() - start
            print(f"done ({elapsed:.2f}s)")

            if use_chat_template:
                messages.append({"role": "assistant", "content": output})
            else:
                conv.messages[-1][-1] = output

    print("Warmup complete. Starting evaluation...")

    # Track statistics
    all_stats = {
        'total_tokens': [],
        'total_iterations': [],
        'wall_time': [],
        'tokens_per_second': [],
    }

    if use_eagle:
        all_stats['accept_lengths'] = []
        all_stats['avg_accept_length'] = []
        all_stats['tokens_per_iter'] = []
        all_stats['speedup'] = []

    # Evaluate questions
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    for question in tqdm(questions, desc="Evaluating"):
        torch.manual_seed(0)
        if use_chat_template:
            messages = []
        else:
            conv = get_conversation_template(conv_template)

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

            if use_eagle:
                output_ids, new_tokens, iterations, accept_lengths, iteration_traces = model.eagenerate(
                    torch.as_tensor(input_ids).to(model.base_model.device),
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    log=True,
                    is_llama3=is_llama3,
                    collect_expert_traces=args.collect_expert_traces,
                )
            else:
                # Baseline: use naivegenerate (optimized AR decoding)
                output_ids, new_tokens, iterations = model.naivegenerate(
                    torch.as_tensor(input_ids).to(model.base_model.device),
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    log=True,
                    is_llama3=is_llama3
                )
                accept_lengths = None

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

            if use_eagle and accept_lengths is not None:
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
                    turn_stats['expert_traces'] = iteration_traces

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
            "model_id": model_id,
            "choices": [{
                "index": 0,
                "turns": turns,
                "stats": turns_stats
            }],
            "tstamp": time.time(),
        }

        with open(answer_file, "a") as fout:
            fout.write(json.dumps(ans_json) + "\n")

        # Aggregate statistics
        for turn_stats in turns_stats:
            all_stats['total_tokens'].append(turn_stats['tokens'])
            all_stats['total_iterations'].append(turn_stats['iterations'])
            all_stats['wall_time'].append(turn_stats['time'])
            all_stats['tokens_per_second'].append(turn_stats['throughput'])

            if use_eagle:
                all_stats['avg_accept_length'].append(turn_stats['avg_accept_length'])
                all_stats['tokens_per_iter'].append(turn_stats['tokens_per_iter'])
                all_stats['speedup'].append(turn_stats['speedup'])

    # Print summary statistics
    print("\n" + "="*70)
    print(f"EVALUATION SUMMARY - {'EAGLE' if use_eagle else 'BASELINE'}")
    print("="*70)
    print(f"Questions evaluated: {len(questions)}")
    print(f"Total turns: {len(all_stats['total_tokens'])}")
    print(f"\nAggregate Statistics:")
    print(f"  Total tokens generated: {sum(all_stats['total_tokens'])}")
    print(f"  Total iterations: {sum(all_stats['total_iterations'])}")
    print(f"  Total time: {sum(all_stats['wall_time']):.2f}s")
    print(f"  Mean throughput: {np.mean(all_stats['tokens_per_second']):.2f} tokens/s")
    print(f"  Median throughput: {np.median(all_stats['tokens_per_second']):.2f} tokens/s")

    if use_eagle:
        print(f"\nEAGLE Statistics:")
        print(f"  Mean acceptance ratio: {np.mean(all_stats['tokens_per_iter']):.2f}")
        print(f"  Total tokens target: {args.total_token}")

    print("="*70)
    print(f"\nResults saved to: {answer_file}")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation script for EAGLE models across multiple benchmarks")

    # Model arguments
    parser.add_argument("--base-model-path", type=str, required=True,
                        help="Path or HF repo ID for base model")
    parser.add_argument("--ea-model-path", type=str, default=None,
                        help="Path or HF repo ID for EAGLE adapter (optional for baseline)")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Model identifier for output file")
    parser.add_argument("--conv-template", type=str, default="vicuna",
                        help="Conversation template (vicuna, llama-2, llama-3, etc.)")

    # Evaluation arguments
    parser.add_argument("--bench-name", type=str, default="mt_bench",
                        help="Benchmark name")
    parser.add_argument("--question-file", type=str, default=None,
                        help="Path to question file (default: auto-detect from bench-name)")
    parser.add_argument("--answer-file", type=str, default=None,
                        help="Path to answer file (default: auto-generate)")
    parser.add_argument("--question-begin", type=int, default=None,
                        help="A debug option. The begin index of questions.")
    parser.add_argument("--question-end", type=int, default=None,
                        help="A debug option. The end index of questions.")
    parser.add_argument("--num-questions", type=int, default=None,
                        help="Number of questions to evaluate (default: all, ignored if question-begin/end used)")

    # Generation arguments
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Maximum number of new tokens")
    parser.add_argument("--warmup-tokens", type=int, default=64,
                        help="Number of tokens to generate during warmup (default: 64)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")

    # EAGLE arguments
    parser.add_argument("--use-eagle", action="store_true",
                        help="Use EAGLE acceleration (requires --ea-model-path)")
    parser.add_argument("--use-eagle3", action="store_true",
                        help="Use EAGLE-3 mode")
    parser.add_argument("--total-token", type=int, default=63,
                        help="Number of draft tokens for EAGLE")
    parser.add_argument("--max-active-experts", type=int, default=None,
                        help="Cap the number of active experts dispatched per wave (default: unlimited)")
    parser.add_argument("--collect-expert-traces", action="store_true",
                        help="Capture per-iteration draft tree and routing traces (adds overhead)")

    args = parser.parse_args()

    # Validate arguments
    if args.use_eagle and not args.ea_model_path:
        parser.error("--use-eagle requires --ea-model-path")

    # Setup paths
    if args.question_file is None:
        args.question_file = os.path.join(parent_dir, f"data/{args.bench_name}/question.jsonl")

    if args.answer_file is None:
        mode_suffix = "eagle" if args.use_eagle else "baseline"
        args.answer_file = f"results/{args.bench_name}/{args.model_id}-{mode_suffix}-t{args.temperature}.jsonl"

    # Print configuration
    print("="*70)
    print("CONFIGURATION:")
    print("="*70)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("="*70)

    # Run evaluation
    run_evaluation(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        model_id=args.model_id,
        conv_template=args.conv_template,
        question_file=args.question_file,
        answer_file=args.answer_file,
        num_questions=args.num_questions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_eagle=args.use_eagle,
        args=args
    )
