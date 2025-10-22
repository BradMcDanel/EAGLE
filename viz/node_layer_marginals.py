#!/usr/bin/env python3
"""Compute per-node marginal expert activations and correlate early vs late layers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd


def _ensure_list(obj: Any) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return list(obj)


def _flatten_numeric_sequence(obj: Any, expected_len: int) -> List[float]:
    """Flatten possibly nested numeric containers into a fixed-length list."""

    result: List[float] = []

    def _recurse(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                _recurse(item)
        elif value is None:
            result.append(math.nan)
        else:
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                result.append(math.nan)

    if obj is not None:
        _recurse(obj)

    if len(result) < expected_len:
        result.extend([math.nan] * (expected_len - len(result)))
    return result[:expected_len]


def _build_layer_node_sets(experts: Dict[str, Any], num_nodes: int) -> Dict[str, List[set[int]]]:
    layer_node_sets: Dict[str, List[set[int]]] = {}
    for layer_key, payload in experts.items():
        layer_data = payload.get("experts") if isinstance(payload, dict) else payload
        if isinstance(layer_data, list) and layer_data and isinstance(layer_data[0], list):
            per_node = layer_data[0]
        else:
            per_node = layer_data
        per_node = _ensure_list(per_node)
        node_sets = [set(map(int, node)) for node in per_node]
        if len(node_sets) != num_nodes:
            # pad or truncate to keep alignment with tree nodes
            if len(node_sets) < num_nodes:
                node_sets.extend([set()] * (num_nodes - len(node_sets)))
            else:
                node_sets = node_sets[:num_nodes]
        layer_node_sets[layer_key] = node_sets
    return layer_node_sets


def _iter_trace_waves(paths: Iterable[Path], labels: Iterable[str]) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for path, label in zip(paths, labels):
        with path.open() as fh:
            for record_index, line in enumerate(fh):
                record = json.loads(line)
                base_meta = {
                    "dataset": label,
                    "trace_file": str(path),
                    "record_index": record_index,
                    "question_id": record.get("question_id"),
                    "answer_id": record.get("answer_id"),
                    "model_id": record.get("model_id"),
                }
                for choice_index, choice in enumerate(record.get("choices", [])):
                    stats_list = choice.get("stats", [])
                    for stats_index, stats in enumerate(stats_list):
                        waves = stats.get("expert_traces", [])
                        for wave_index, wave in enumerate(waves):
                            meta = {
                                **base_meta,
                                "choice_index": choice_index,
                                "stats_index": stats_index,
                                "wave_index": wave_index,
                            }
                            yield meta, wave


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


def _rows_for_wave(
    meta: Dict[str, Any],
    wave: Dict[str, Any],
    layer_keys: List[str],
    layer_node_sets: Dict[str, List[set[int]]],
    node_order: List[int],
    parents: List[int],
    depths: List[int],
    tokens: List[Any],
    routing_weights: List[float],
    subtree_weights: List[float],
    accepted_nodes: set[int],
    accepted_order: Dict[int, int],
    children: List[List[int]],
    subtree_sizes: List[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cumulative: Dict[int, Dict[str, set[int]]] = {}
    node_log_probs = wave.get("node_log_probs", {}) or {}
    draft_log_probs_map = wave.get("draft_node_log_probs", {}) or {}

    tree_total_nodes = len(parents)
    tree_depth = max(depths) if depths else -1
    accepted_len = wave.get("accepted_length")
    tree_width = _ensure_list(wave.get("tree_width_by_depth"))

    for node_idx in node_order:
        parent_idx = parents[node_idx]
        parent_cum = cumulative.get(parent_idx, {}) if parent_idx >= 0 else {}

        cumulative[node_idx] = {}
        early_sum = 0
        late_sum = 0
        total_sum = 0

        depth_val = depths[node_idx] if node_idx < len(depths) else math.nan
        width_val = (
            tree_width[depth_val]
            if isinstance(depth_val, int) and 0 <= depth_val < len(tree_width)
            else math.nan
        )

        row = {
            "dataset": meta["dataset"],
            "trace_file": meta["trace_file"],
            "record_index": meta["record_index"],
            "question_id": meta.get("question_id"),
            "answer_id": meta.get("answer_id"),
            "model_id": meta.get("model_id"),
            "choice_index": meta["choice_index"],
            "stats_index": meta["stats_index"],
            "wave_index": meta["wave_index"],
            "iteration": wave.get("iteration", meta["wave_index"]),
            "node": node_idx,
            "parent": parent_idx,
            "depth": depth_val,
            "token_id": tokens[node_idx] if node_idx < len(tokens) else None,
            "accepted": int(node_idx in accepted_nodes),
            "accepted_order": accepted_order.get(node_idx, -1),
            "children_count": len(children[node_idx]) if node_idx < len(children) else 0,
            "is_leaf": int(len(children[node_idx]) == 0) if node_idx < len(children) else 0,
            "is_root": int(parent_idx < 0),
            "subtree_size": subtree_sizes[node_idx] if node_idx < len(subtree_sizes) else math.nan,
            "tree_total_nodes": tree_total_nodes,
            "tree_depth": tree_depth,
            "tree_width_at_depth": width_val,
            "accepted_length": accepted_len,
            "routing_weight": routing_weights[node_idx] if node_idx < len(routing_weights) else math.nan,
            "subtree_weight": subtree_weights[node_idx] if node_idx < len(subtree_weights) else math.nan,
            "log_prob": float(node_log_probs.get(str(node_idx), math.nan)),
            "draft_log_prob": float(draft_log_probs_map.get(str(node_idx), math.nan)),
        }

        for layer_key in layer_keys:
            node_set = layer_node_sets[layer_key][node_idx]
            parent_set = parent_cum.get(layer_key, set())
            marginal = node_set - parent_set
            cumulative[node_idx][layer_key] = parent_set.union(node_set)

            marginal_count = len(marginal)
            cumulative_count = len(cumulative[node_idx][layer_key])

            row[f"layer_{layer_key}_marginal"] = marginal_count
            row[f"layer_{layer_key}_cumulative"] = cumulative_count

            layer_idx = int(layer_key)
            if layer_idx <= 2:
                early_sum += marginal_count
            if layer_idx >= 12:
                late_sum += marginal_count
            total_sum += marginal_count

            if layer_key == "0":
                row["layer0_set"] = " ".join(str(e) for e in sorted(node_set))

        row["early_marginal"] = early_sum
        row["late_marginal"] = late_sum
        row["total_marginal"] = total_sum

        rows.append(row)

    return rows


def parse_trace_files(paths: Iterable[Path], labels: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict] = []

    for meta, wave in _iter_trace_waves(paths, labels):
        experts = wave.get("experts") or {}
        if not experts:
            continue

        parents = _ensure_list(wave.get("parents"))
        depths = _ensure_list(wave.get("depth"))
        if not parents or not depths or len(parents) != len(depths):
            continue

        num_nodes = len(parents)
        layer_node_sets = _build_layer_node_sets(experts, num_nodes)
        if not layer_node_sets:
            continue

        layer_keys = sorted(layer_node_sets.keys(), key=int)
        node_order = sorted(range(num_nodes), key=lambda idx: depths[idx])

        tokens = _ensure_list(wave.get("tokens"))
        if len(tokens) < num_nodes:
            tokens = tokens + [None] * (num_nodes - len(tokens))
        else:
            tokens = tokens[:num_nodes]

        routing_weights = _flatten_numeric_sequence(wave.get("routing_weights"), num_nodes)
        subtree_weights = _flatten_numeric_sequence(wave.get("subtree_weights"), num_nodes)

        accepted_sequence = _ensure_list(wave.get("accepted_nodes"))
        accepted_nodes = set(int(n) for n in accepted_sequence)
        accepted_order = {int(node): idx for idx, node in enumerate(accepted_sequence)}

        children = _compute_children(parents, num_nodes)
        reverse_order = sorted(node_order, key=lambda idx: depths[idx], reverse=True)
        subtree_sizes = _compute_subtree_sizes(children, reverse_order)

        rows.extend(
            _rows_for_wave(
                meta=meta,
                wave=wave,
                layer_keys=layer_keys,
                layer_node_sets=layer_node_sets,
                node_order=node_order,
                parents=parents,
                depths=depths,
                tokens=tokens,
                routing_weights=routing_weights,
                subtree_weights=subtree_weights,
                accepted_nodes=accepted_nodes,
                accepted_order=accepted_order,
                children=children,
                subtree_sizes=subtree_sizes,
            )
        )
    return pd.DataFrame(rows)


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    corrs = []
    for dataset, group in df.groupby("dataset"):
        if group["early_marginal"].std() == 0 or group["late_marginal"].std() == 0:
            value = float("nan")
        else:
            value = float(group["early_marginal"].corr(group["late_marginal"]))
        corrs.append({"dataset": dataset, "corr_early_late": value})
    return pd.DataFrame(corrs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-node early vs late layer marginal correlation")
    parser.add_argument("trace_files", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/combined/node_layer_marginals.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/combined/node_layer_marginal_summary.csv"))
    args = parser.parse_args()

    labels = args.labels or [path.parent.name for path in args.trace_files]
    if len(labels) != len(args.trace_files):
        raise ValueError("Number of labels must match number of trace files")

    df = parse_trace_files(args.trace_files, labels)
    if df.empty:
        print("No per-node expert data found")
        return

    df.to_csv(args.out, index=False)
    summary = compute_correlations(df)
    summary.to_csv(args.summary, index=False)
    print(summary)


if __name__ == "__main__":
    main()
