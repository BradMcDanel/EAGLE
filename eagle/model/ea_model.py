import copy
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
try:
    from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
except ImportError:
    KVQwen3ForCausalLM = None
try:
    from .modeling_qwen3_moe_kv import KVQwen3MoeForCausalLM
except ImportError:
    KVQwen3MoeForCausalLM = None
from .modeling_olmoe_kv import OlmoeForCausalLM as KVOlmoeForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig


class ExpertTraceCollector:
    """Collects per-layer expert routing data for a single forward pass."""

    def __init__(self) -> None:
        self.records: Dict[int, Dict[str, List]] = {}

    def record(
        self,
        layer_idx: int,
        experts: torch.Tensor,
        weights: torch.Tensor,
        shortlist: Optional[List[int]] = None,
        full_experts: Optional[torch.Tensor] = None,
        full_weights: Optional[torch.Tensor] = None,
        precap_shortlist: Optional[List[int]] = None,
    ) -> None:
        experts_cpu = experts.to(torch.int64).cpu()
        weights_cpu = weights.to(torch.float32).cpu()
        record: Dict[str, Any] = {
            "experts": experts_cpu.tolist(),
            "weights": weights_cpu.tolist(),
        }
        if shortlist is not None:
            record["shortlist"] = list(shortlist)
        if full_experts is not None:
            record["full_experts"] = full_experts.to(torch.int64).cpu().tolist()
        if full_weights is not None:
            record["full_weights"] = full_weights.to(torch.float32).cpu().tolist()
        if precap_shortlist is not None:
            record["precap_shortlist"] = list(precap_shortlist)
        self.records[int(layer_idx)] = record

    def to_serializable(self) -> Dict[str, Dict[str, List]]:
        return {str(k): v for k, v in sorted(self.records.items())}


class OracleTracePruner:
    """Replay oracle traces to prune draft trees with perfect knowledge."""

    def __init__(
        self,
        trace_map: Dict[Tuple[str, int], List[List[int]]],
        source: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        self.trace_map = trace_map
        self.source = source
        self.strict = bool(strict)
        self.active: bool = False
        self.current_key: Optional[Tuple[str, int]] = None
        self.current_iterations: List[Dict[str, Any]] = []
        self.cursor: int = 0
        self.current_turn_stats: Dict[str, Any] = {}
        self.current_iteration_summaries: List[Dict[str, Any]] = []

    @classmethod
    def from_trace_file(
        cls,
        path: str,
        choice_index: int = 0,
        strict: bool = False,
    ) -> "OracleTracePruner":
        trace_map: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        with open(path, "r") as f:
            raw_content = f.read()

        records: List[Dict[str, Any]] = []
        try:
            parsed = json.loads(raw_content)
            if isinstance(parsed, list):
                records = [item for item in parsed if isinstance(item, dict)]
            elif isinstance(parsed, dict):
                records = [parsed]
        except json.JSONDecodeError:
            for line in raw_content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)

        for record in records:
            question_id = record.get("question_id")
            key_prefix = str(question_id)
            choices = record.get("choices", [])
            if not choices:
                continue
            if not (0 <= choice_index < len(choices)):
                continue
            choice = choices[choice_index]
            stats = choice.get("stats") or []
            for turn_idx, stat in enumerate(stats):
                traces = stat.get("expert_traces") or []
                if not traces:
                    continue
                turn_key = (key_prefix, int(turn_idx))
                accepted_sequences: List[Dict[str, Any]] = []
                for trace in traces:
                    nodes = trace.get("accepted_nodes") or []
                    cleaned: List[int] = []
                    for node in nodes:
                        if isinstance(node, int):
                            cleaned.append(node)
                        else:
                            try:
                                cleaned.append(int(node))
                            except (TypeError, ValueError):
                                continue
                    tokens = trace.get("tokens") or []
                    token_path: List[int] = []
                    for node_idx in cleaned:
                        if 0 <= node_idx < len(tokens):
                            token_path.append(int(tokens[node_idx]))
                        else:
                            token_path.append(-1)
                    accepted_sequences.append({"nodes": cleaned, "tokens": token_path})
                if accepted_sequences:
                    trace_map[turn_key] = accepted_sequences
        return cls(trace_map=trace_map, source=path, strict=strict)

    def has_trace_for(self, question_id: Any, turn_index: int) -> bool:
        key = (str(question_id), int(turn_index))
        return key in self.trace_map

    def start_turn(self, question_id: Any, turn_index: int) -> bool:
        key = (str(question_id), int(turn_index))
        self.current_key = key
        self.current_iterations = [dict(seq) for seq in self.trace_map.get(key, [])]
        self.cursor = 0
        self.active = bool(self.current_iterations)
        self.current_iteration_summaries = []
        self.current_turn_stats = {
            "question_id": question_id,
            "turn_index": int(turn_index),
            "expected_iterations": len(self.current_iterations),
            "applied_iterations": 0,
            "missing_iterations": 0,
            "fallback_iterations": 0,
            "paths_total": 0,
            "paths_kept": 0,
            "source": self.source,
        }
        if not self.active and self.strict:
            raise KeyError(
                f"No oracle trace available for question={question_id!r}, turn={turn_index}"
            )
        return self.active

    def finish_turn(self) -> Optional[Dict[str, Any]]:
        if not self.current_turn_stats:
            self.active = False
            self.current_key = None
            self.current_iterations = []
            self.cursor = 0
            return None

        expected = self.current_turn_stats.get("expected_iterations", 0)
        applied = self.current_turn_stats.get("applied_iterations", 0)
        missing = self.current_turn_stats.get("missing_iterations", 0)
        self.current_turn_stats["unused_iterations"] = max(expected - applied, 0)
        self.current_turn_stats["status"] = (
            "ok" if applied == expected and missing == 0 else "partial"
        )
        summary = dict(self.current_turn_stats)

        if self.strict and summary["status"] != "ok":
            raise RuntimeError(
                f"Oracle trace replay incomplete for key={self.current_key}, "
                f"summary={summary}"
            )

        self.active = False
        self.current_key = None
        self.current_iterations = []
        self.cursor = 0
        self.current_iteration_summaries = []
        self.current_turn_stats = {}
        return summary

    def apply(
        self,
        iteration_index: int,
        retrieve_indices: torch.Tensor,
        parents: List[int],
        tokens: List[int],
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]], Optional[List[bool]], Optional[torch.Tensor]]:
        if not self.active:
            return retrieve_indices, None, None, None

        summary: Dict[str, Any] = {
            "iteration": int(iteration_index),
            "status": "ok",
        }

        if self.cursor >= len(self.current_iterations):
            summary["status"] = "exhausted"
            summary["paths_total"] = int(retrieve_indices.size(0)) if retrieve_indices.dim() > 0 else 0
            summary["paths_kept"] = summary["paths_total"]
            summary["fallback_used"] = 0
            self.current_turn_stats["missing_iterations"] += 1
            self.current_iteration_summaries.append(summary)
            return retrieve_indices, summary, None, None

        expected_record = self.current_iterations[self.cursor]
        self.cursor += 1
        self.current_turn_stats["applied_iterations"] += 1

        num_nodes = len(parents)
        expected_nodes_raw = [int(node) for node in expected_record.get("nodes", [])]
        expected_tokens = [int(tok) for tok in expected_record.get("tokens", [])]
        summary["expected_nodes"] = expected_nodes_raw
        summary["expected_tokens"] = expected_tokens

        if not expected_tokens:
            raise RuntimeError("Oracle replay: expected token path is empty")

        retrieve_cpu = retrieve_indices.detach().to("cpu")
        tokens_len = len(expected_tokens)
        matched_nodes: Optional[List[int]] = None
        matched_row: Optional[int] = None

        if tokens is None or len(tokens) == 0:
            raise RuntimeError("Oracle replay: current tree tokens unavailable")

        for row_idx in range(retrieve_cpu.size(0)):
            nodes = [int(val) for val in retrieve_cpu[row_idx].tolist() if val >= 0]
            if len(nodes) < tokens_len:
                continue
            row_tokens: List[int] = []
            valid = True
            for node in nodes[:tokens_len]:
                if 0 <= node < len(tokens):
                    row_tokens.append(int(tokens[node]))
                else:
                    valid = False
                    break
            if not valid:
                continue
            if row_tokens == expected_tokens:
                matched_nodes = nodes[:tokens_len]
                matched_row = row_idx
                break

        if matched_nodes is None or matched_row is None:
            raise RuntimeError(
                f"Oracle replay: no tree path matches expected tokens {expected_tokens}"
            )

        keep_mask = [False] * num_nodes
        for node in matched_nodes:
            if 0 <= node < num_nodes:
                keep_mask[node] = True
        if num_nodes > 0:
            keep_mask[0] = True

        summary["nodes_kept"] = int(sum(keep_mask))
        summary["matched_nodes"] = matched_nodes
        summary["matched_tokens"] = expected_tokens
        summary["paths_total"] = retrieve_cpu.size(0)
        summary["paths_kept"] = 1
        summary["kept_rows"] = [int(matched_row)]

        filtered_indices = retrieve_indices.index_select(
            0, torch.tensor([matched_row], dtype=torch.long, device=retrieve_indices.device)
        )
        kept_rows = torch.tensor([matched_row], dtype=torch.long, device=retrieve_indices.device)

        self.current_turn_stats["paths_total"] += summary.get("paths_total", 0)
        self.current_turn_stats["paths_kept"] += summary.get("paths_kept", 0)
        self.current_iteration_summaries.append(summary)
        return filtered_indices, summary, keep_mask, kept_rows


def _prune_tree_to_path(
    draft_tokens: torch.Tensor,
    tree_parents: torch.Tensor,
    tree_position_ids: torch.Tensor,
    draft_log_probs: torch.Tensor,
    keep_nodes: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Rebuild draft tree tensors so only the oracle path remains."""

    if not keep_nodes:
        raise RuntimeError("Oracle replay: keep_nodes is empty")

    device = draft_tokens.device
    keep_order: List[int] = []
    seen: set[int] = set()
    for node in keep_nodes:
        if node < 0 or node in seen:
            continue
        keep_order.append(int(node))
        seen.add(int(node))

    if not keep_order:
        raise RuntimeError("Oracle replay: no valid nodes to keep")

    if keep_order[0] != 0:
        keep_order.insert(0, 0)

    tree_parents_flat = tree_parents.view(-1)
    depth_flat = tree_position_ids.view(-1)
    keep_order = sorted(keep_order, key=lambda idx: (int(depth_flat[idx].item()), int(idx)))

    index_map: Dict[int, int] = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_order)}
    num_nodes = len(keep_order)

    # Remap parents
    new_parents: List[int] = []
    for old_idx in keep_order:
        parent = int(tree_parents_flat[old_idx].item())
        if parent < 0:
            new_parents.append(-1)
        elif parent in index_map:
            new_parents.append(index_map[parent])
        else:
            raise RuntimeError(
                f"Oracle replay: parent node {parent} missing from keep set {keep_nodes}"
            )

    # Compute new depths and validate structure
    new_depths: List[int] = []
    for idx, parent in enumerate(new_parents):
        if parent < 0:
            new_depths.append(0)
        else:
            new_depths.append(new_depths[parent] + 1)

    depth_to_node: Dict[int, int] = {}
    for node_idx, depth in enumerate(new_depths):
        if depth in depth_to_node:
            raise RuntimeError(
                f"Oracle replay: multiple nodes share depth {depth} in path {keep_order}"
            )
        depth_to_node[depth] = node_idx

    max_depth = max(new_depths)
    expected_depths = set(range(max_depth + 1))
    if set(depth_to_node.keys()) != expected_depths:
        raise RuntimeError(
            f"Oracle replay: depth coverage mismatch (expected {expected_depths}, got {set(depth_to_node.keys())})"
        )

    index_tensor = torch.tensor(keep_order, dtype=torch.long, device=device)
    new_draft_tokens = draft_tokens.index_select(1, index_tensor)
    new_draft_log_probs = draft_log_probs.index_select(0, index_tensor)

    new_parents_tensor = torch.tensor(new_parents, dtype=tree_parents.dtype, device=device)
    new_depths_tensor = torch.tensor(new_depths, dtype=tree_position_ids.dtype, device=device)

    mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
    for idx in range(num_nodes):
        current = idx
        while current >= 0:
            mask[idx, current] = True
            current = new_parents[current]

    new_tree_mask = mask.float()[None, None]
    new_tree_position_ids = new_depths_tensor.unsqueeze(0)

    retrieve_row = torch.full((max_depth + 1,), -1, dtype=torch.long, device=device)
    for depth in range(max_depth + 1):
        retrieve_row[depth] = depth_to_node[depth]
    new_retrieve_indices = retrieve_row.unsqueeze(0)

    return (
        new_draft_tokens,
        new_tree_mask,
        new_tree_position_ids,
        new_parents_tensor,
        new_draft_log_probs,
        new_retrieve_indices,
        new_depths,
        keep_order,
    )


def set_expert_trace_recorder(model: nn.Module, collector: Optional[ExpertTraceCollector]) -> int:
    """Attach or clear expert trace recorders across all supported MoE blocks."""

    updated = 0
    recorder = collector.record if collector is not None else None
    for module in model.modules():
        setter = getattr(module, "set_trace_recorder", None)
        if callable(setter):
            setter(recorder)
            updated += 1
    return updated


def compute_subtree_weights(tree_parents: torch.Tensor, tree_position_ids: torch.Tensor) -> torch.Tensor:
    """Return subtree weights for each node (broadcasting over batch if needed)."""

    if tree_parents.dim() == 1:
        return _compute_single_subtree_weights(tree_parents, tree_position_ids)

    weights = []
    for b in range(tree_parents.size(0)):
        weights.append(_compute_single_subtree_weights(tree_parents[b], tree_position_ids[b]))
    return torch.stack(weights, dim=0)


def _compute_single_subtree_weights(tree_parents: torch.Tensor, tree_position_ids: torch.Tensor) -> torch.Tensor:
    num_nodes = tree_parents.numel()
    weights = torch.ones(num_nodes, dtype=torch.float32, device=tree_parents.device)

    depths = tree_position_ids.to(torch.long)
    order = torch.argsort(depths, descending=True)
    for idx in order.tolist():
        parent = int(tree_parents[idx].item())
        if parent >= 0:
            weights[parent] += weights[idx]

    weights /= weights.sum()
    return weights


def summarize_layer_experts(
    layer_record: Dict[str, List],
    node_depths: List[int],
    accepted_nodes: set,
) -> Dict[str, Any]:
    """Build per-layer aggregate statistics over expert routes."""

    experts_list = layer_record.get("experts", []) or []
    weights_list = layer_record.get("weights", []) or []

    def _collapse_batch_axis(values: List) -> List:
        if not isinstance(values, list):
            return []
        if not values:
            return []
        first = values[0]
        if isinstance(first, list):
            if len(values) == 1:
                return first
            flattened: List = []
            for item in values:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            return flattened
        return values

    def _flatten_numbers(values) -> List[float]:
        result: List[float] = []
        if isinstance(values, (list, tuple)):
            for item in values:
                result.extend(_flatten_numbers(item))
        elif values is not None:
            try:
                result.append(float(values))
            except (TypeError, ValueError):
                pass
        return result

    per_node_experts = _collapse_batch_axis(experts_list)
    per_node_weights = _collapse_batch_axis(weights_list)
    full_experts_lists = _collapse_batch_axis(layer_record.get("full_experts"))
    full_weights_lists = _collapse_batch_axis(layer_record.get("full_weights"))
    precap_shortlist = layer_record.get("precap_shortlist") or []

    depth_unique: Dict[int, set] = defaultdict(set)
    accepted_depth_unique: Dict[int, set] = defaultdict(set)
    depth_weight: Dict[int, float] = defaultdict(float)
    accepted_depth_weight: Dict[int, float] = defaultdict(float)

    total_unique: set = set()
    precap_unique: set = set()
    accepted_unique: set = set()
    total_weight = 0.0
    accepted_weight = 0.0

    for node_idx, node_experts in enumerate(per_node_experts):
        if node_experts is None:
            continue

        flat_experts = _flatten_numbers(node_experts)
        if not flat_experts:
            continue

        experts_int = [int(e) for e in flat_experts]
        total_unique.update(experts_int)

        node_depth = node_depths[node_idx] if node_idx < len(node_depths) else None
        if node_depth is not None:
            depth_unique[node_depth].update(experts_int)

        if node_idx in accepted_nodes:
            accepted_unique.update(experts_int)
            if node_depth is not None:
                accepted_depth_unique[node_depth].update(experts_int)

        node_weights = per_node_weights[node_idx] if node_idx < len(per_node_weights) else []
        flat_weights = _flatten_numbers(node_weights)
        if flat_weights:
            weight_sum = float(sum(flat_weights[: len(experts_int)]))
            total_weight += weight_sum
            if node_depth is not None:
                depth_weight[node_depth] += weight_sum
            if node_idx in accepted_nodes:
                accepted_weight += weight_sum
                if node_depth is not None:
                    accepted_depth_weight[node_depth] += weight_sum

        if node_idx < len(full_experts_lists):
            flat_full = _flatten_numbers(full_experts_lists[node_idx])
            if flat_full:
                precap_unique.update(int(e) for e in flat_full)

    precap_unique.update(int(idx) for idx in precap_shortlist)

    result = {
        "total_unique": len(total_unique),
        "accepted_unique": len(accepted_unique),
        "total_unique_ids": sorted(total_unique),
        "accepted_unique_ids": sorted(accepted_unique),
        "depth_unique": {str(k): len(v) for k, v in depth_unique.items() if v},
        "accepted_depth_unique": {str(k): len(v) for k, v in accepted_depth_unique.items() if v},
        "depth_weight": {str(k): depth_weight[k] for k in depth_weight},
        "accepted_depth_weight": {str(k): accepted_depth_weight[k] for k in accepted_depth_weight},
        "total_weight": total_weight,
        "accepted_weight": accepted_weight,
    }
    if precap_unique:
        result["precap_unique"] = len(precap_unique)
        result["precap_unique_ids"] = sorted(precap_unique)
    return result


def _compute_children(parents: List[int], num_nodes: int) -> List[List[int]]:
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    for node_idx, parent_idx in enumerate(parents):
        if 0 <= parent_idx < num_nodes:
            children[parent_idx].append(node_idx)
    return children


def _compute_subtree_sizes(children: List[List[int]], traversal: List[int]) -> List[int]:
    sizes = [1] * len(children)
    for node_idx in traversal:
        for child_idx in children[node_idx]:
            sizes[node_idx] += sizes[child_idx]
    return sizes


def _normalize_int_lists(payload: Any, num_nodes: int) -> List[List[int]]:
    values = payload or []
    if (
        isinstance(values, list)
        and len(values) == 1
        and isinstance(values[0], list)
        and len(values[0]) == num_nodes
    ):
        values = values[0]
    result: List[List[int]] = []
    for idx in range(num_nodes):
        if isinstance(values, list) and idx < len(values) and isinstance(values[idx], list):
            entry = [
                int(v)
                for v in values[idx]
                if isinstance(v, (int, float))
            ]
            result.append(entry)
        else:
            result.append([])
    return result


def _normalize_float_lists(payload: Any, num_nodes: int) -> List[List[float]]:
    values = payload or []
    if (
        isinstance(values, list)
        and len(values) == 1
        and isinstance(values[0], list)
        and len(values[0]) == num_nodes
    ):
        values = values[0]
    result: List[List[float]] = []
    for idx in range(num_nodes):
        if isinstance(values, list) and idx < len(values) and isinstance(values[idx], list):
            entry = [
                float(v)
                for v in values[idx]
                if isinstance(v, (int, float))
            ]
            result.append(entry)
        else:
            result.append([])
    return result


def _compute_layer0_stats(
    collector: Optional["ExpertTraceCollector"],
    num_nodes: int,
) -> Dict[str, Any]:
    experts_per_node = [[] for _ in range(num_nodes)]
    routing_weights = [math.nan] * num_nodes
    if collector is None:
        return {
            "experts": experts_per_node,
            "routing_weights": routing_weights,
        }
    layer_record = collector.records.get(0) if hasattr(collector, "records") else None
    if layer_record is None:
        return {
            "experts": experts_per_node,
            "routing_weights": routing_weights,
        }
    experts_per_node = _normalize_int_lists(layer_record.get("experts"), num_nodes)
    weight_lists = _normalize_float_lists(layer_record.get("weights"), num_nodes)
    routing_weights = [
        float(sum(weights)) if weights else math.nan
        for weights in weight_lists
    ]
    return {
        "experts": experts_per_node,
        "routing_weights": routing_weights,
    }


def _build_training_iteration_trace(
    iteration_index: int,
    parents: List[int],
    depths: List[int],
    tokens: List[int],
    accepted_path: List[int],
    depth_hist: List[int],
    tree_depth: int,
    subtree_weights_payload: Optional[List[Any]],
    node_log_probs: Dict[str, float],
    draft_log_probs: Dict[str, float],
    collector: Optional["ExpertTraceCollector"],
) -> Dict[str, Any]:
    num_nodes = len(parents)
    accepted_nodes = [node for node in accepted_path if node >= 0]
    accepted_set = set(accepted_nodes)
    accepted_order = {node: idx for idx, node in enumerate(accepted_nodes)}

    # Tree structure helpers
    children = _compute_children(parents, num_nodes)
    reverse_order = sorted(range(num_nodes), key=lambda idx: depths[idx], reverse=True)
    subtree_sizes = _compute_subtree_sizes(children, reverse_order)

    # Layer-0 routing features
    layer0_stats = _compute_layer0_stats(collector, num_nodes)
    layer0_expert_lists: List[List[int]] = layer0_stats["experts"]
    routing_weight_sums: List[float] = layer0_stats["routing_weights"]

    layer0_sets = [set(node_experts) for node_experts in layer0_expert_lists]

    # Gather per-layer expert sets (layer 0 plus higher layers)
    layer_node_sets: Dict[int, List[set[int]]] = {0: layer0_sets}
    if collector is not None:
        for layer_idx, layer_record in collector.records.items():
            if layer_idx == 0:
                continue
            expert_lists = _normalize_int_lists(layer_record.get("experts"), num_nodes)
            layer_node_sets[layer_idx] = [set(items) for items in expert_lists]

    node_order = sorted(range(num_nodes), key=lambda idx: depths[idx])
    layer_cumulative_sets: Dict[int, List[set[int]]] = {
        layer_idx: [set() for _ in range(num_nodes)]
        for layer_idx in layer_node_sets
    }
    layer_marginal_counts: Dict[int, List[int]] = {
        layer_idx: [0] * num_nodes
        for layer_idx in layer_node_sets
    }
    layer_cumulative_counts: Dict[int, List[int]] = {
        layer_idx: [0] * num_nodes
        for layer_idx in layer_node_sets
    }
    layer0_set_strings: List[str] = [""] * num_nodes
    wave_layer_unique_sets: Dict[int, set[int]] = {layer_idx: set() for layer_idx in layer_node_sets}

    for node_idx in node_order:
        parent_idx = parents[node_idx]
        for layer_idx, node_sets in layer_node_sets.items():
            node_set = node_sets[node_idx]
            if 0 <= parent_idx < num_nodes:
                parent_set = layer_cumulative_sets[layer_idx][parent_idx]
            else:
                parent_set = set()
            marginal = node_set - parent_set
            cumulative = parent_set.union(node_set)
            layer_cumulative_sets[layer_idx][node_idx] = cumulative
            layer_marginal_counts[layer_idx][node_idx] = len(marginal)
            layer_cumulative_counts[layer_idx][node_idx] = len(cumulative)
            if layer_idx == 0:
                layer0_set_strings[node_idx] = " ".join(str(e) for e in sorted(node_set)) if node_set else ""
            wave_layer_unique_sets[layer_idx].update(node_set)

    layer0_marginal = layer_marginal_counts.get(0, [0] * num_nodes)
    layer0_cumulative = layer_cumulative_counts.get(0, [0] * num_nodes)
    layer0_unique = [len(s) for s in layer_node_sets.get(0, [set() for _ in range(num_nodes)])]

    total_marginal: List[int] = [0] * num_nodes
    for layer_idx, counts in layer_marginal_counts.items():
        for node_idx, value in enumerate(counts):
            total_marginal[node_idx] += value

    higher_marginal: List[int] = [total_marginal[idx] - layer0_marginal[idx] for idx in range(num_nodes)]

    layer_unique_totals: Dict[int, int] = {layer_idx: len(expert_set) for layer_idx, expert_set in wave_layer_unique_sets.items()}
    wave_total_unique = sum(layer_unique_totals.values())
    wave_layer0_unique = layer_unique_totals.get(0, 0)
    wave_higher_unique = max(wave_total_unique - wave_layer0_unique, 0)

    # Subtree weights
    if isinstance(subtree_weights_payload, list) and subtree_weights_payload and isinstance(subtree_weights_payload[0], list):
        subtree_weights = subtree_weights_payload[0]
    else:
        subtree_weights = subtree_weights_payload or []
    if len(subtree_weights) < num_nodes:
        subtree_weights = subtree_weights + [math.nan] * (num_nodes - len(subtree_weights))
    else:
        subtree_weights = subtree_weights[:num_nodes]

    # Ensure token list matches node count
    if len(tokens) < num_nodes:
        tokens = tokens + [None] * (num_nodes - len(tokens))
    else:
        tokens = tokens[:num_nodes]

    accepted_length = len(accepted_set)
    accepted_layer_sets: Dict[int, set[int]] = {layer_idx: set() for layer_idx in layer_node_sets}
    for node_idx in accepted_set:
        if node_idx < num_nodes:
            for layer_idx, node_sets in layer_node_sets.items():
                if node_idx < len(node_sets):
                    accepted_layer_sets[layer_idx].update(node_sets[node_idx])

    layer_accepted_totals: Dict[int, int] = {
        layer_idx: len(expert_set) for layer_idx, expert_set in accepted_layer_sets.items()
    }
    wave_accepted_unique = sum(layer_accepted_totals.values())

    node_rows: List[Dict[str, Any]] = []
    for node_idx in range(num_nodes):
        depth_val = depths[node_idx] if node_idx < len(depths) else -1
        width_at_depth = depth_hist[depth_val] if 0 <= depth_val < len(depth_hist) else math.nan
        early_sum = 0
        late_sum = 0
        row = {
            "node": node_idx,
            "parent": parents[node_idx],
            "depth": depth_val,
            "token_id": tokens[node_idx],
            "accepted": 1 if node_idx in accepted_set else 0,
            "accepted_order": accepted_order.get(node_idx, -1),
            "children_count": len(children[node_idx]),
            "is_leaf": 1 if not children[node_idx] else 0,
            "is_root": 1 if parents[node_idx] < 0 else 0,
            "subtree_size": subtree_sizes[node_idx],
            "tree_total_nodes": num_nodes,
            "tree_depth": tree_depth,
            "tree_width_at_depth": width_at_depth,
            "routing_weight": routing_weight_sums[node_idx] if node_idx < len(routing_weight_sums) else math.nan,
            "subtree_weight": subtree_weights[node_idx] if node_idx < len(subtree_weights) else math.nan,
            "draft_log_prob": float(draft_log_probs.get(str(node_idx), math.nan)),
            "layer_0_marginal": layer0_marginal[node_idx],
            "layer_0_cumulative": layer0_cumulative[node_idx],
            "layer0_set": layer0_set_strings[node_idx],
            "layer0_unique": layer0_unique[node_idx],
            "total_marginal": total_marginal[node_idx],
            "higher_marginal": higher_marginal[node_idx],
        }
        for layer_idx, counts in layer_marginal_counts.items():
            count_val = counts[node_idx]
            row[f"layer_{layer_idx}_marginal"] = count_val
            row[f"layer_{layer_idx}_cumulative"] = layer_cumulative_counts[layer_idx][node_idx]
            if layer_idx <= 2:
                early_sum += count_val
            if layer_idx >= 12:
                late_sum += count_val
        row["early_marginal"] = early_sum
        row["late_marginal"] = late_sum
        row["wave_total_unique"] = wave_total_unique
        row["wave_layer0_unique"] = wave_layer0_unique
        row["wave_higher_unique"] = wave_higher_unique
        row["wave_accepted_unique"] = wave_accepted_unique
        for layer_idx, total_val in layer_unique_totals.items():
            row[f"wave_layer_{layer_idx}_total_unique"] = total_val
        for layer_idx, accepted_val in layer_accepted_totals.items():
            row[f"wave_layer_{layer_idx}_accepted_unique"] = accepted_val
        node_rows.append(row)

    return {
        "schema": "training",
        "iteration": iteration_index,
        "tree_stats": {
            "total_nodes": num_nodes,
            "depth": tree_depth,
            "width_by_depth": depth_hist,
        },
        "post_verify": {
            "accepted_nodes": accepted_nodes,
            "accepted_length": accepted_length,
            "node_log_probs": node_log_probs,
        },
        "node_features": node_rows,
    }


def _build_analysis_iteration_trace(
    iteration_index: int,
    tokens: List[int],
    parents: List[int],
    depths: List[int],
    retrieve_indices: List[List[int]],
    accepted_path: List[int],
    tree_depth: int,
    depth_hist: List[int],
    total_nodes: int,
    accepted_length: int,
    collector: Optional["ExpertTraceCollector"],
    subtree_weights_cpu: Optional[List[Any]],
    verification_info: Dict[str, Any],
    node_log_probs: Dict[str, float],
    draft_log_probs_dict: Dict[str, float],
) -> Dict[str, Any]:
    accepted_nodes_set = {n for n in accepted_path if n >= 0}
    accepted_experts_summary: Dict[str, Any] = {}
    layer_expert_stats: Dict[str, Any] = {}
    combined_total_unique = 0
    combined_accepted_unique = 0
    node_depths_list = depths
    if collector is not None:
        for layer_idx, layer_record in collector.records.items():
            stats = summarize_layer_experts(layer_record, node_depths_list, accepted_nodes_set)
            layer_key = str(layer_idx)
            layer_expert_stats[layer_key] = stats
            accepted_experts_summary[layer_key] = {
                "unique_count": stats["accepted_unique"],
                "experts": stats["accepted_unique_ids"],
            }
            combined_total_unique += stats["total_unique"]
            combined_accepted_unique += stats["accepted_unique"]

    pruned_nodes = total_nodes - accepted_length

    return {
        "schema": "analysis",
        "iteration": iteration_index,
        "tokens": tokens,
        "parents": parents,
        "depth": depths,
        "retrieve_indices": retrieve_indices,
        "accepted_nodes": accepted_path,
        "tree_total_nodes": total_nodes,
        "tree_depth": tree_depth,
        "tree_width_by_depth": depth_hist,
        "accepted_length": accepted_length,
        "accepted_unique_experts": accepted_experts_summary,
        "layer_expert_stats": layer_expert_stats,
        "total_unique_experts": combined_total_unique,
        "accepted_unique_experts_count": combined_accepted_unique,
        "pruned_nodes": pruned_nodes,
        "subtree_weights": subtree_weights_cpu,
        "verification": verification_info,
        "node_log_probs": node_log_probs,
        "draft_node_log_probs": draft_log_probs_dict,
        "experts": collector.to_serializable() if collector is not None else {},
    }


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, trust_remote_code=True)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()
        self.oracle: Optional[OracleTracePruner] = None
        self._expert_cap: Optional[int] = None
        self._expert_cap_config: Optional[Dict[str, Any]] = None
        self._expert_pruning_strategy: str = "substitution"
        self._cap_suspend_depth: int = 0
        self._last_cap_usage_means: List[float] = []
        self._apply_expert_caps()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            if KVQwen3ForCausalLM is None:
                raise ImportError("Qwen3ForCausalLM support is unavailable in this build of transformers")
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3MoeForCausalLM':
            if KVQwen3MoeForCausalLM is None:
                raise ImportError("Qwen3MoeForCausalLM support is unavailable in this build of transformers")
            base_model = KVQwen3MoeForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'OlmoeForCausalLM':
            base_model = KVOlmoeForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def configure_oracle(
            self,
            trace_file: Optional[str] = None,
            choice_index: int = 0,
            strict: bool = False,
    ) -> bool:
        if trace_file is None:
            self.oracle = None
            return False
        if not os.path.exists(trace_file):
            raise FileNotFoundError(f"Oracle trace file not found: {trace_file}")
        pruner = OracleTracePruner.from_trace_file(
            path=trace_file,
            choice_index=choice_index,
            strict=strict,
        )
        if not pruner.trace_map and strict:
            raise ValueError(f"No oracle traces loaded from {trace_file}")
        self.oracle = pruner
        return bool(pruner.trace_map)

    def _set_all_expert_caps(self, cap: Optional[int], config: Optional[Dict[str, Any]]) -> None:
        effective_cap = None if cap is None else int(cap)
        for module in self.base_model.modules():
            if hasattr(module, "expert_cap"):
                module.expert_cap = effective_cap
            if hasattr(module, "expert_cap_config"):
                module.expert_cap_config = None if config is None else dict(config)
            if hasattr(module, "expert_pruning_strategy"):
                module.expert_pruning_strategy = self._expert_pruning_strategy

    def _apply_expert_caps(self) -> None:
        if self._cap_suspend_depth > 0:
            return
        self._set_all_expert_caps(self._expert_cap, self._expert_cap_config)

    def set_expert_pruning_strategy(self, strategy: str) -> None:
        """Set expert pruning strategy: 'substitution' or 'truncation'."""
        assert strategy in ["substitution", "truncation"], f"Invalid pruning strategy: {strategy}"
        self._expert_pruning_strategy = strategy
        if self._cap_suspend_depth == 0:
            self._apply_expert_caps()

    def set_expert_count_budget(self, budget: int) -> None:
        """Set cardinality-based expert selection (fixed number of experts)."""
        assert budget > 0, f"Expert count budget must be positive, got {budget}"
        self._expert_cap = int(budget)
        self._expert_cap_config = None
        if self._cap_suspend_depth == 0:
            self._apply_expert_caps()

    def set_probability_expert_selection(self, target: float) -> None:
        """Set probability-based expert selection (preserve routing probability mass)."""
        assert 0 < target <= 1, f"Probability budget target must be in (0, 1], got {target}"
        config: Dict[str, Any] = {
            "mode": "probability",
            "target": float(target),
        }
        self._expert_cap = None
        self._expert_cap_config = config
        if self._cap_suspend_depth == 0:
            self._apply_expert_caps()

    @contextmanager
    def suspend_expert_cap(self):
        """Temporarily disable expert capping (e.g., during KV prefill)."""
        if self._expert_cap is None and not self._expert_cap_config:
            yield
            return
        if self._cap_suspend_depth == 0:
            self._set_all_expert_caps(None, None)
        self._cap_suspend_depth += 1
        try:
            yield
        finally:
            self._cap_suspend_depth -= 1
            if self._cap_suspend_depth == 0:
                self._apply_expert_caps()

    def has_oracle(self) -> bool:
        return self.oracle is not None

    def has_active_oracle(self) -> bool:
        return bool(self.oracle and self.oracle.active)

    def start_oracle_turn(self, question_id: Any, turn_index: int) -> bool:
        if self.oracle is None:
            return False
        return self.oracle.start_turn(question_id, turn_index)

    def finish_oracle_turn(self) -> Optional[Dict[str, Any]]:
        if self.oracle is None:
            return None
        return self.oracle.finish_turn()

    def pop_last_cap_usage_means(self) -> List[float]:
        data = list(self._last_cap_usage_means)
        self._last_cap_usage_means = []
        return data

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            output_hidden_states: bool = False,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=output_hidden_states,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            collect_expert_traces=False,
            trace_schema: str = "analysis",
            adaptive_controller: Optional[Any] = None,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        trace_schema = (trace_schema or "analysis").lower()
        if trace_schema not in {"analysis", "training"}:
            trace_schema = "analysis"

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()
        original_cap = getattr(self, "_expert_cap", None)
        pending_cap: Optional[int] = None
        if adaptive_controller is not None:
            on_start = getattr(adaptive_controller, "on_sequence_start", None)
            if callable(on_start):
                on_start()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # prefill
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
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        accept_lengths = []  # Track acceptance lengths per iteration
        iteration_traces: List[Dict[str, Any]] = []
        max_length = max_length - self.ea_layer.total_tokens - 10
        iteration_cap_means: List[float] = []
        for idx in range(max_length):
            if pending_cap is not None:
                self.set_expert_cap(pending_cap)
                pending_cap = None
            cap_for_iteration = getattr(self, "_expert_cap", None)
            tracker_reset = getattr(self.base_model.model, "reset_cap_usage_tracker", None)
            if callable(tracker_reset):
                tracker_reset()
            iter_start_time = time.time()
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            iteration_oracle_summary: Optional[Dict[str, Any]] = None
            iter_draft_tokens = None
            iter_tree_parents = None
            iter_tree_position_ids = None
            iter_retrieve_indices = None
            iter_draft_log_probs = None
            depth_hist: List[int] = []
            tree_depth = 0
            total_nodes = 0

            nodes_before_prune = int(tree_parents.view(-1).numel())

            nodes_before_prune = int(tree_parents.view(-1).numel())

            # Snapshot current tree before verification (CPU copies for logging)
            oracle_active = self.has_active_oracle()
            need_wave_snapshot = (
                collect_expert_traces or (trace_schema == "training") or oracle_active
            )
            if need_wave_snapshot:
                iter_draft_tokens = draft_tokens.detach().to("cpu")
                iter_tree_parents = tree_parents.detach().to("cpu")
                iter_tree_position_ids = tree_position_ids.detach().to("cpu")
                iter_retrieve_indices = retrieve_indices.detach().to("cpu")
                iter_draft_log_probs = draft_log_probs.detach().to("cpu")
                iter_tree_depths = iter_tree_position_ids.view(-1)
                tree_depth = int(iter_tree_depths.max().item()) if iter_tree_depths.numel() else 0
                depth_hist = torch.bincount(iter_tree_depths, minlength=tree_depth + 1).tolist()
                total_nodes = int(iter_tree_depths.numel())

            if oracle_active and self.oracle is not None:
                parents_list_for_oracle: List[int]
                if iter_tree_parents is not None:
                    parents_list_for_oracle = iter_tree_parents.view(-1).tolist()
                else:
                    parents_list_for_oracle = tree_parents.detach().to("cpu").view(-1).tolist()

                if iter_draft_tokens is not None:
                    tokens_for_oracle = iter_draft_tokens[0].tolist()
                else:
                    tokens_for_oracle = draft_tokens.detach().to("cpu")[0].tolist()

                _, iteration_oracle_summary, keep_mask, _ = self.oracle.apply(
                    iteration_index=idx,
                    retrieve_indices=retrieve_indices,
                    parents=parents_list_for_oracle,
                    tokens=tokens_for_oracle,
                )

                keep_sequence = []
                if iteration_oracle_summary is not None:
                    keep_sequence = [
                        int(node)
                        for node in iteration_oracle_summary.get("matched_nodes", [])
                        if isinstance(node, int) and node >= 0
                    ]
                if not keep_sequence and keep_mask is not None:
                    keep_sequence = [i for i, flag in enumerate(keep_mask) if flag]

                (
                    draft_tokens,
                    tree_mask,
                    tree_position_ids,
                    tree_parents,
                    draft_log_probs,
                    retrieve_indices,
                    _new_depths_list,
                    keep_order,
                ) = _prune_tree_to_path(
                    draft_tokens,
                    tree_parents,
                    tree_position_ids,
                    draft_log_probs,
                    keep_sequence,
                )

                nodes_after_prune = int(draft_tokens.size(1))
                if iteration_oracle_summary is not None:
                    iteration_oracle_summary["nodes_before_prune"] = nodes_before_prune
                    iteration_oracle_summary["nodes_after_prune"] = nodes_after_prune
                    iteration_oracle_summary["nodes_pruned"] = max(nodes_before_prune - nodes_after_prune, 0)
                    iteration_oracle_summary["path_length"] = int(len(keep_order))

                self.base_model.model.tree_mask = tree_mask
                nodes_before_prune = nodes_after_prune

                if need_wave_snapshot:
                    iter_draft_tokens = draft_tokens.detach().to("cpu")
                    iter_tree_parents = tree_parents.detach().to("cpu")
                    iter_tree_position_ids = tree_position_ids.detach().to("cpu")
                    iter_retrieve_indices = retrieve_indices.detach().to("cpu")
                    iter_draft_log_probs = draft_log_probs.detach().to("cpu")
                    iter_tree_depths = iter_tree_position_ids.view(-1)
                    tree_depth = int(iter_tree_depths.max().item()) if iter_tree_depths.numel() else 0
                    depth_hist = torch.bincount(iter_tree_depths, minlength=tree_depth + 1).tolist()
                    total_nodes = int(iter_tree_depths.numel())

            draft_tokens = draft_tokens.to(input_ids.device)
            tree_mask = tree_mask.to(input_ids.device)
            tree_position_ids = tree_position_ids.to(input_ids.device)
            tree_parents = tree_parents.to(input_ids.device)
            draft_log_probs = draft_log_probs.to(input_ids.device)
            retrieve_indices = retrieve_indices.to(input_ids.device)
            # Target model forward, get logits
            collector = (
                ExpertTraceCollector()
                if (collect_expert_traces or trace_schema == "training")
            else None
            )
            if collector is not None:
                set_expert_trace_recorder(self.base_model.model, collector)

            subtree_weights_cpu = None
            try:
                subtree_weights = compute_subtree_weights(tree_parents, tree_position_ids)
                if subtree_weights is not None:
                    if subtree_weights.dim() == 1:
                        subtree_weights = subtree_weights.unsqueeze(0)
                    subtree_weights = subtree_weights.to(draft_tokens.device)
                    if need_wave_snapshot:
                        subtree_weights_cpu = subtree_weights.detach().to("cpu").tolist()
            except Exception:
                subtree_weights_cpu = None

            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
                draft_log_probs,
            )

            if collector is not None:
                set_expert_trace_recorder(self.base_model.model, None)

            draft_log_probs_dict: Dict[str, float] = {}
            draft_log_probs_list: List[float] = []
            parents_list: List[int] = []
            depths_list: List[int] = []
            tokens_list: List[int] = []
            if need_wave_snapshot and iter_draft_log_probs is not None:
                try:
                    draft_log_probs_list = iter_draft_log_probs.tolist()
                    draft_log_probs_dict = {str(idx): float(val) for idx, val in enumerate(draft_log_probs_list)}
                except Exception:
                    draft_log_probs_dict = {}
                parents_list = iter_tree_parents.view(-1).tolist()
                depths_list = iter_tree_position_ids.view(-1).tolist()
                tokens_list = iter_draft_tokens[0].tolist()

            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices.to(draft_tokens.device)]
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            best_candidate_val = (
                int(best_candidate.item()) if isinstance(best_candidate, torch.Tensor) else int(best_candidate)
            )
            accept_length_val = (
                int(accept_length.item()) if isinstance(accept_length, torch.Tensor) else int(accept_length)
            )
            accept_lengths.append(accept_length_val)
            margin_val = 0.0
            try:
                candidate_logits = logits.detach()
                if candidate_logits.dim() == 3:
                    pos = max(0, min(accept_length_val, candidate_logits.size(1) - 1))
                    candidate_logits = candidate_logits[best_candidate_val, pos]
                elif candidate_logits.dim() == 2:
                    candidate_logits = candidate_logits[best_candidate_val]
                if candidate_logits.dim() == 1 and candidate_logits.numel() >= 2:
                    top_vals, _ = torch.topk(candidate_logits.float(), k=2)
                    margin_val = float((top_vals[0] - top_vals[1]).item())
                elif candidate_logits.dim() == 1 and candidate_logits.numel() == 1:
                    margin_val = float(candidate_logits.item())
            except Exception:
                margin_val = 0.0
            verification_info: Dict[str, Any] = {}
            if collect_expert_traces:
                strategy = "greedy" if logits_processor is None else "posterior"
                logits_cpu = logits.detach().to("cpu")
                logits_norm = float(torch.linalg.vector_norm(logits_cpu).item())
                posterior_slice = logits_cpu[best_candidate_val, : max(1, accept_length_val + 1)]
                verification_info = {
                    "strategy": strategy,
                    "best_candidate": best_candidate_val,
                    "accept_length": accept_length_val,
                    "logits_norm": logits_norm,
                    "posterior_slice": posterior_slice.tolist(),
                }

            if collect_expert_traces:
                accepted_path = iter_retrieve_indices[best_candidate_val, : accept_length_val + 1].tolist()

                iter_tree_depths = iter_tree_position_ids.view(-1)
                tree_depth = int(iter_tree_depths.max().item()) if iter_tree_depths.numel() else 0
                depth_hist = torch.bincount(iter_tree_depths, minlength=tree_depth + 1).tolist()
                total_nodes = int(iter_tree_depths.numel())

                node_log_probs: Dict[str, float] = {}
                try:
                    log_probs = torch.log_softmax(logits, dim=-1).detach().to("cpu")
                    if log_probs.dim() == 3:
                        num_cands, seq_len, _ = log_probs.shape
                        for cand_idx in range(num_cands):
                            for pos in range(1, iter_retrieve_indices.size(1)):
                                node_idx = int(iter_retrieve_indices[cand_idx, pos].item())
                                if node_idx < 0:
                                    continue
                                if node_idx >= iter_draft_tokens.size(1):
                                    continue
                                token_id = int(iter_draft_tokens[0, node_idx].item())
                                if token_id < 0 or pos - 1 >= seq_len:
                                    continue
                                logp = float(log_probs[cand_idx, pos - 1, token_id].item())
                                node_log_probs.setdefault(str(node_idx), logp)
                except Exception:
                    node_log_probs = {}

                draft_log_probs_dict: Dict[str, float] = {}
                try:
                    draft_log_probs_list = iter_draft_log_probs.tolist()
                    draft_log_probs_dict = {
                        str(idx): float(val) for idx, val in enumerate(draft_log_probs_list)
                    }
                except Exception:
                    draft_log_probs_dict = {}

                parents_list = iter_tree_parents.view(-1).tolist()
                depths_list = iter_tree_depths.tolist()
                tokens_list = iter_draft_tokens[0].tolist()
                retrieve_list = iter_retrieve_indices.tolist()
                accepted_length = len({n for n in accepted_path if n >= 0})

                if trace_schema == "training":
                    iteration_record = _build_training_iteration_trace(
                        idx,
                        parents_list,
                        depths_list,
                        tokens_list,
                        accepted_path,
                        depth_hist,
                        tree_depth,
                        subtree_weights_cpu,
                        node_log_probs,
                        draft_log_probs_dict,
                        collector,
                    )
                else:
                    iteration_record = _build_analysis_iteration_trace(
                        idx,
                        tokens_list,
                        parents_list,
                        depths_list,
                        retrieve_list,
                        accepted_path,
                        tree_depth,
                        depth_hist,
                        total_nodes,
                        accepted_length,
                        collector,
                        subtree_weights_cpu,
                        verification_info,
                        node_log_probs,
                        draft_log_probs_dict,
                    )

                if iteration_oracle_summary is not None:
                    iteration_record["oracle"] = iteration_oracle_summary

                iteration_traces.append(iteration_record)
            # Adjusting the input sequence, draft model forward
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                tree_parents,
                draft_log_probs,
                new_token,
                hidden_state,
                sample_token,
            ) = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            if adaptive_controller is not None:
                iter_elapsed = time.time() - iter_start_time
                iteration_metrics = {
                    "accept_length": accept_length_val,
                    "iteration_time": iter_elapsed,
                    "cap": None if cap_for_iteration is None else int(cap_for_iteration),
                    "margin": margin_val,
                }
                on_iter_end = getattr(adaptive_controller, "on_iteration_end", None)
                if callable(on_iter_end):
                    requested_cap = on_iter_end(idx, iteration_metrics)
                    if requested_cap is not None:
                        pending_cap = int(requested_cap)
            tracker_finalize = getattr(self.base_model.model, "finalize_cap_usage", None)
            if callable(tracker_finalize):
                cap_mean = tracker_finalize()
                if cap_mean is not None:
                    iteration_cap_means.append(float(cap_mean))

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if adaptive_controller is not None:
            finish_cb = getattr(adaptive_controller, "on_sequence_end", None)
            if callable(finish_cb):
                finish_cb()
        if original_cap != getattr(self, "_expert_cap", None):
            self.set_expert_cap(original_cap)
        self._last_cap_usage_means = iteration_cap_means
        if not log:
            return input_ids
        else:
            if collect_expert_traces:
                set_expert_trace_recorder(self.base_model.model, None)
            return input_ids, new_token, idx, accept_lengths, iteration_traces

    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        with self.suspend_expert_cap():
            outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, tree_parents, draft_log_probs, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
                draft_log_probs,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices.to(draft_tokens.device)]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, tree_parents, draft_log_probs, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        with self.suspend_expert_cap():
            outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
