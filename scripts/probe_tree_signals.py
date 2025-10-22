#!/usr/bin/env python3
"""Quick instrumentation script to inspect early-layer signals for OLMoE branches.

Runs a handful of EAGLE draft/verify iterations on a single GSM8K question, capturing
per-candidate logits from both the final layer and an early decoder layer alongside
MoE routing entropy. The goal is to gauge whether we can prune unlikely branches early.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import math

import torch

from fastchat.llm_judge.common import load_questions

# Allow running from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eagle.model.ea_model import EaModel, ExpertTraceCollector, set_expert_trace_recorder
from eagle.model.kv_cache import initialize_past_key_values
from eagle.model.utils import (
    initialize_tree,
    reset_tree_mode,
    tree_decoding,
    update_inference_inputs,
    evaluate_posterior,
)


DEFAULT_BASE_MODEL = "allenai/OLMoE-1B-7B-0125-Instruct"
DEFAULT_EA_MODEL = "wantsleep/OLMoE_1B_7B_Eagle3"
DEFAULT_QUESTION_FILE = Path("eagle/data/gsm8k/question.jsonl")


def build_prompt(tokenizer, turns: Sequence[str]) -> List[int]:
    """Format the GSM8K prompt using the tokenizer's chat template."""

    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer([prompt], add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        raise ValueError("Expected batch size 1 after tokenization")
    return token_ids[0]


def load_first_question(path: Path) -> Dict:
    questions = load_questions(str(path), 0, None)
    if not questions:
        raise ValueError(f"No questions found at {path}")
    return questions[0]


def entropy(weights: torch.Tensor) -> float:
    probs = weights / (weights.sum() + 1e-8)
    log_probs = torch.log(probs + 1e-8)
    return float(-(probs * log_probs).sum().item())


class LayerProbe:
    """Forward hook that captures decoder layer outputs."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.hidden: torch.Tensor | None = None

    def __call__(self, module, inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        self.hidden = hidden.detach()

    def clear(self) -> None:
        self.hidden = None


def compute_candidate_metrics(
    head: torch.nn.Module,
    layer_hidden_map: Dict[int, torch.Tensor],
    logits: torch.Tensor,
    retrieve_indices: torch.Tensor,
    candidates: torch.Tensor,
    collector: ExpertTraceCollector | None = None,
) -> Dict[str, Dict]:
    if logits.dim() == 4:
        logits_tensor = logits.squeeze(0)
    else:
        logits_tensor = logits

    device = logits_tensor.device
    retrieve_indices_dev = retrieve_indices.to(device)
    candidates_dev = candidates.to(device)
    valid_mask = retrieve_indices_dev.ge(0)

    num_candidates, max_len = retrieve_indices_dev.shape

    log_probs_final = torch.full((num_candidates, max_len), float("nan"), device=device)
    if valid_mask.any():
        log_probs_all = torch.log_softmax(logits_tensor, dim=-1)
        gathered = torch.gather(
            log_probs_all,
            dim=2,
            index=candidates_dev.unsqueeze(-1).clamp_min(0),
        ).squeeze(-1)
        log_probs_final = gathered.masked_fill(~valid_mask, float("nan"))

    last_pos = valid_mask.sum(dim=1) - 1
    candidate_scores = torch.full((num_candidates,), float("-inf"), device=device)
    for idx in range(num_candidates):
        pos = last_pos[idx].item()
        if pos >= 0:
            candidate_scores[idx] = log_probs_final[idx, pos]

    layer_logprobs: Dict[int, torch.Tensor] = {}
    layer_candidate_scores: Dict[int, torch.Tensor] = {}

    head_weight = head.weight
    head_bias = head.bias if head.bias is not None else None

    for layer_idx, hidden in layer_hidden_map.items():
        hidden_device = hidden.device
        gather_indices = retrieve_indices.to(hidden_device)
        gathered = hidden[:, gather_indices]
        gathered = gathered.squeeze(0)
        flat = gathered.view(-1, gathered.size(-1))

        valid_flat = valid_mask.to(hidden_device).view(-1)
        token_flat = candidates.to(hidden_device).view(-1).clamp_min(0)

        log_prob_flat = torch.full((flat.size(0),), float("nan"), device=hidden_device, dtype=torch.float32)
        if valid_flat.any():
            selected_hidden = flat[valid_flat]
            logits_flat = torch.matmul(selected_hidden, head_weight.t())
            if head_bias is not None:
                logits_flat += head_bias
            logsumexp = torch.logsumexp(logits_flat, dim=-1)
            token_indices = token_flat[valid_flat].long()
            token_logits = logits_flat[torch.arange(token_indices.size(0), device=hidden_device), token_indices]
            log_prob_selected = (token_logits - logsumexp).to(torch.float32)
            log_prob_flat[valid_flat] = log_prob_selected

        layer_logprob_matrix = log_prob_flat.view(num_candidates, max_len)
        layer_logprobs[layer_idx] = layer_logprob_matrix

        scores = torch.full((num_candidates,), float("-inf"), device=hidden_device)
        for idx in range(num_candidates):
            pos = last_pos[idx].item()
            if pos >= 0:
                scores[idx] = layer_logprob_matrix[idx, pos]
        layer_candidate_scores[layer_idx] = scores

    moe_stats: Dict[str, List[float]] = {"entropy": [], "max_weight": []}
    if collector is not None and collector.records:
        retrieve_indices_cpu = retrieve_indices.to("cpu")
        last_pos_cpu = last_pos.to("cpu")
        for cand_idx in range(num_candidates):
            final_pos = int(last_pos_cpu[cand_idx].item())
            if final_pos < 0:
                moe_stats["entropy"].append(float("nan"))
                moe_stats["max_weight"].append(float("nan"))
                continue
            entropy_acc = 0.0
            max_weight_acc = 0.0
            count = 0
            for record in collector.records.values():
                weights = torch.tensor(record["weights"], dtype=torch.float32)
                weights = weights[0]
                if final_pos >= weights.shape[0]:
                    continue
                token_weights = weights[final_pos]
                entropy_acc += entropy(token_weights)
                max_weight_acc += float(token_weights.max().item())
                count += 1
            if count:
                moe_stats["entropy"].append(entropy_acc / count)
                moe_stats["max_weight"].append(max_weight_acc / count)
            else:
                moe_stats["entropy"].append(float("nan"))
                moe_stats["max_weight"].append(float("nan"))

    return {
        "candidate_scores": candidate_scores.to("cpu"),
        "layer_candidate_scores": {k: v.to("cpu") for k, v in layer_candidate_scores.items()},
        "final_logprob_matrix": log_probs_final.to("cpu"),
        "layer_logprob_matrix": {k: v.to("cpu") for k, v in layer_logprobs.items()},
        "valid_mask": valid_mask.to("cpu"),
        "tokens": candidates.to("cpu"),
        "moe": moe_stats,
    }


def run_probe(
    model: EaModel,
    input_ids: List[int],
    max_iterations: int,
    layer_indices: List[int],
    temperature: float = 0.0,
) -> None:
    device = model.base_model.lm_head.weight.device
    padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(device)
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    model.ea_layer.reset_kv()

    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        current_length_data.zero_()
    else:
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
            model.base_model, max_length=2048
        )
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    reset_tree_mode(model)

    (
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        tree_parents,
        draft_log_probs,
        logits,
        hidden_state,
        sample_token,
    ) = initialize_tree(
        input_ids_tensor,
        model,
        past_key_values,
        logits_processor=None,
    )

    num_layers = model.base_model.config.num_hidden_layers
    layer_indices = sorted({max(0, min(num_layers - 1, idx)) for idx in layer_indices})

    probes: Dict[int, LayerProbe] = {}
    handles = []
    for layer_idx in layer_indices:
        probe = LayerProbe(layer_idx)
        handle = model.base_model.model.layers[layer_idx].register_forward_hook(probe)
        probes[layer_idx] = probe
        handles.append(handle)

    new_token_count = 0

    for iteration in range(max_iterations):
        model.base_model.model.tree_mask = tree_mask

        for probe in probes.values():
            probe.clear()

        collector = ExpertTraceCollector()
        set_expert_trace_recorder(model.base_model.model, collector)

        logits_batch, hidden_state_new, outputs = tree_decoding(
            model,
            draft_tokens,
            past_key_values,
            tree_position_ids,
            input_ids_tensor,
            retrieve_indices,
        )
        set_expert_trace_recorder(model.base_model.model, None)

        draft_tokens = torch.cat((draft_tokens, padding), dim=1)
        candidates = draft_tokens[0, retrieve_indices]

        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits_batch,
            candidates,
            logits_processor=None,
        )

        best_idx = int(best_candidate.item()) if torch.is_tensor(best_candidate) else int(best_candidate)
        accept_len_val = int(accept_length)

        layer_hidden_map: Dict[int, torch.Tensor] = {}
        for layer_idx, probe in probes.items():
            if probe.hidden is None:
                raise RuntimeError(f"Layer {layer_idx} did not produce hidden states")
            layer_hidden_map[layer_idx] = probe.hidden.to(model.base_model.lm_head.weight.device)

        metrics = compute_candidate_metrics(
            model.base_model.lm_head,
            layer_hidden_map,
            logits_batch,
            retrieve_indices,
            candidates,
            collector,
        )
        candidate_scores = metrics["candidate_scores"]
        layer_candidate_scores = metrics["layer_candidate_scores"]
        final_logprob_matrix = metrics["final_logprob_matrix"]
        layer_logprob_matrix = metrics["layer_logprob_matrix"]
        valid_mask = metrics["valid_mask"]
        tokens_matrix = metrics["tokens"]

        print(f"\nIteration {iteration}")
        print(f"  verifier best candidate: {best_idx}")
        for layer_idx in layer_indices:
            layer_scores = layer_candidate_scores.get(layer_idx)
            if layer_scores is None:
                continue
            layer_best = int(torch.argmax(layer_scores).item())
            match = layer_best == best_idx
            gap = candidate_scores[best_idx] - layer_scores[best_idx]
            print(f"  layer {layer_idx:02d}: best={layer_best} match={match} gap={gap:.3f}")

        if iteration == 0:
            print_node_alignment(
                model,
                retrieve_indices,
                tokens_matrix,
                valid_mask,
                final_logprob_matrix,
                layer_logprob_matrix,
                layer_indices,
                best_idx,
                accept_len_val,
                tree_position_ids,
            )

        # Advance to next iteration
        (
            input_ids_tensor,
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            tree_parents,
            draft_log_probs,
            new_token_count,
            _,
            sample_token,
        ) = update_inference_inputs(
            input_ids_tensor,
            candidates,
            best_candidate,
            accept_len_val,
            retrieve_indices,
            logits_processor=None,
            new_token=new_token_count,
            past_key_values_data_list=past_key_values_data,
            current_length_data=current_length_data,
            model=model,
            hidden_state_new=hidden_state_new,
            sample_p=sample_p,
        )

        if iteration + 1 >= max_iterations:
            break

    for handle in handles:
        handle.remove()


def _format_token(tokenizer, token_id: int) -> str:
    if token_id < 0:
        return "<pad>"
    text = tokenizer.decode([int(token_id)])
    text = text.replace("\n", "\\n")
    return text if text else "<empty>"


def _safe_float(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return f"{value}"
    return f"{value:.3f}"


def print_node_alignment(
    model: EaModel,
    retrieve_indices: torch.Tensor,
    tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    final_logprob_matrix: torch.Tensor,
    layer_logprob_matrix: Dict[int, torch.Tensor],
    layer_indices: List[int],
    best_candidate: int,
    accept_length: int,
    tree_position_ids: torch.Tensor,
    sample_limit: int = 6,
) -> None:
    tokenizer = model.tokenizer

    retrieve_cpu = retrieve_indices.to("cpu")
    tokens_cpu = tokens.to("cpu")
    valid_cpu = valid_mask.to("cpu")
    final_lp_cpu = final_logprob_matrix.to("cpu")
    layer_lp_cpu = {k: v.to("cpu") for k, v in layer_logprob_matrix.items()}

    try:
        depth_matrix = tree_position_ids.to("cpu")[retrieve_cpu]
    except Exception:
        depth_matrix = tree_position_ids.to("cpu").unsqueeze(0)[..., retrieve_cpu]

    accepted_mask = torch.zeros_like(valid_cpu, dtype=torch.bool)
    if 0 <= best_candidate < accepted_mask.shape[0]:
        end = min(accept_length + 1, accepted_mask.shape[1])
        accepted_mask[best_candidate, :end] = True

    def describe_node(c_idx: int, step: int) -> str:
        node_id = int(retrieve_cpu[c_idx, step].item())
        token_id = int(tokens_cpu[c_idx, step].item())
        depth = int(depth_matrix[c_idx, step]) if depth_matrix.ndim == 2 else int(depth_matrix[0, c_idx, step])
        pieces = [
            f"cand={c_idx:02d}",
            f"step={step:02d}",
            f"node={node_id:03d}",
            f"depth={depth:02d}",
            f"tok={token_id}",
            f"text={_format_token(tokenizer, token_id)!r}",
            f"final={_safe_float(float(final_lp_cpu[c_idx, step]))}",
        ]
        for layer_idx in layer_indices:
            layer_val = layer_lp_cpu.get(layer_idx)
            if layer_val is not None:
                pieces.append(f"L{layer_idx:02d}={_safe_float(float(layer_val[c_idx, step]))}")
        return " ".join(pieces)

    print("    Accepted prefix nodes:")
    for pos in range(accepted_mask.shape[1]):
        if not accepted_mask[best_candidate, pos]:
            continue
        print("      " + describe_node(best_candidate, pos))

    print("    Sample rejected nodes:")
    shown = 0
    for cand_idx in range(retrieve_cpu.shape[0]):
        for pos in range(retrieve_cpu.shape[1]):
            if not valid_cpu[cand_idx, pos]:
                continue
            if accepted_mask[cand_idx, pos]:
                continue
            print("      " + describe_node(cand_idx, pos))
            shown += 1
            if shown >= sample_limit:
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe early-layer signals for OLMoE EAGLE runs")
    parser.add_argument("--iterations", type=int, default=3, help="Number of draft/verify iterations to run")
    parser.add_argument(
        "--layers",
        type=str,
        default="2,8,16",
        help="Comma-separated decoder layer indices to probe (0-based)",
    )
    parser.add_argument(
        "--question-file",
        type=Path,
        default=DEFAULT_QUESTION_FILE,
        help="GSM8K question file (jsonl)",
    )
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--ea-model", type=str, default=DEFAULT_EA_MODEL)
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g., cuda:0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(0)

    question = load_first_question(args.question_file)

    print(f"Loaded question id={question.get('question_id', 'unknown')}")

    model = EaModel.from_pretrained(
        use_eagle3=True,
        base_model_path=args.base_model,
        ea_model_path=args.ea_model,
        total_token=63,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if args.device is None else None,
    )
    if args.device is not None:
        model = model.to(args.device)

    layer_indices = [int(item) for item in args.layers.split(",") if item.strip()]
    if not layer_indices:
        layer_indices = [0]

    input_ids = build_prompt(model.tokenizer, question["turns"])
    run_probe(
        model,
        input_ids,
        max_iterations=args.iterations,
        layer_indices=layer_indices,
    )


if __name__ == "__main__":
    main()
