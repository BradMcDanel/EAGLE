#!/usr/bin/env python3
"""Analyze draft-tree shapes vs. MoE expert usage from EAGLE trace files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


def load_traces(paths: Sequence[Path], labels: Sequence[str], layers: List[int], max_depth_cols: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for path, label in zip(paths, labels):
        with path.open() as fh:
            for line in fh:
                record = json.loads(line)
                for choice in record.get("choices", []):
                    for turn_stats in choice.get("stats", []):
                        traces = turn_stats.get("expert_traces", [])
                        for wave in traces:
                            row: Dict = {
                                "dataset": label,
                                "iteration": wave.get("iteration"),
                                "nodes": wave.get("tree_total_nodes"),
                                "depth": wave.get("tree_depth"),
                                "accepted_len": wave.get("accepted_length"),
                                "pruned_nodes": wave.get("pruned_nodes"),
                                "accepted_unique_total": wave.get("accepted_unique_experts_count"),
                                "total_unique_all": wave.get("total_unique_experts"),
                            }
                            width_hist = wave.get("tree_width_by_depth", []) or []
                            for idx in range(min(max_depth_cols, len(width_hist))):
                                row[f"width_depth_{idx}"] = width_hist[idx]

                            layer_stats = wave.get("layer_expert_stats", {}) or {}
                            accepted_experts = wave.get("accepted_unique_experts", {}) or {}
                            for layer in layers:
                                key = str(layer)
                                stats = layer_stats.get(key, {})
                                row[f"layer_{layer}_accepted_unique"] = stats.get("accepted_unique", 0)
                                row[f"layer_{layer}_total_unique"] = stats.get("total_unique", 0)
                                row[f"layer_{layer}_accepted_weight"] = stats.get("accepted_weight", 0.0)
                                row[f"layer_{layer}_total_weight"] = stats.get("total_weight", 0.0)
                                # Back-compat fallback if summary missing
                                if stats.get("accepted_unique") is None:
                                    layer_info = accepted_experts.get(key, {})
                                    layer_experts = layer_info.get("experts", [])
                                    row[f"layer_{layer}_accepted_unique"] = len(layer_experts)

                            if layer_stats:
                                row["accepted_unique_total"] = sum(
                                    stats.get("accepted_unique", 0) for stats in layer_stats.values()
                                )
                                row["total_unique_all"] = sum(
                                    stats.get("total_unique", 0) for stats in layer_stats.values()
                                )
                                row["layer_count"] = len(layer_stats)
                            else:
                                row["layer_count"] = 0

                            verification = wave.get("verification", {}) or {}
                            row["verification_strategy"] = verification.get("strategy")
                            row["verification_accept_length"] = verification.get("accept_length")

                            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, layers: List[int], max_depth_cols: int) -> pd.DataFrame:
    grouped = df.groupby(["dataset", "depth"], dropna=False)
    summary = grouped.agg(
        nodes_mean=("nodes", "mean"),
        accepted_len_mean=("accepted_len", "mean"),
        pruned_nodes_mean=("pruned_nodes", "mean"),
        accepted_unique_mean=("accepted_unique_total", "mean"),
        total_unique_mean=("total_unique_all", "mean"),
        count=("iteration", "count"),
    )
    for layer in layers:
        summary[f"layer_{layer}_accepted_unique_mean"] = grouped[f"layer_{layer}_accepted_unique"].mean()
        summary[f"layer_{layer}_total_unique_mean"] = grouped[f"layer_{layer}_total_unique"].mean()
        if f"layer_{layer}_accepted_weight" in df:
            summary[f"layer_{layer}_accepted_weight_mean"] = grouped[f"layer_{layer}_accepted_weight"].mean()
        if f"layer_{layer}_total_weight" in df:
            summary[f"layer_{layer}_total_weight_mean"] = grouped[f"layer_{layer}_total_weight"].mean()
    for idx in range(max_depth_cols):
        col = f"width_depth_{idx}"
        if col in df:
            summary[f"{col}_mean"] = grouped[col].mean()
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tree shape vs. expert usage")
    parser.add_argument("trace_files", nargs="+", type=Path)
    parser.add_argument("--layers", nargs="*", type=int, default=[0], help="MoE layer indices to report")
    parser.add_argument("--max-depth-columns", type=int, default=6, help="Number of depth levels to record widths for")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV summary output path")
    parser.add_argument("--raw", type=Path, default=None, help="Optional CSV with per-wave raw data")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional dataset labels matching the trace files",
    )
    args = parser.parse_args()

    labels = args.labels or [path.parent.name for path in args.trace_files]
    if len(labels) != len(args.trace_files):
        raise ValueError("Number of labels must match number of trace files")

    df = load_traces(args.trace_files, labels, args.layers, args.max_depth_columns)
    if df.empty:
        print("No iteration_traces found in input files")
        return

    print(f"Loaded {len(df)} draft/verify cycles from {len(args.trace_files)} trace files")
    summary = summarize(df, args.layers, args.max_depth_columns)
    print(summary)

    if args.raw:
        df.to_csv(args.raw, index=False)
        print("Wrote raw data to", args.raw)

    if args.out:
        summary.to_csv(args.out, index=False)
        print("Wrote summary to", args.out)


if __name__ == "__main__":
    main()
