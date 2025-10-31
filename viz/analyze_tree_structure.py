#!/usr/bin/env python3
"""Profile draft-tree shapes and relate them to expert usage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _to_list(obj):
    if isinstance(obj, list):
        return obj
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return list(obj)


def extract_cycles(trace_paths: Iterable[Path], labels: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_depth_rows: List[Dict] = []
    cycle_rows: List[Dict] = []

    for path, label in zip(trace_paths, labels):
        with path.open() as fh:
            for line in fh:
                record = json.loads(line)
                for choice in record.get("choices", []):
                    for stats in choice.get("stats", []):
                        for wave in stats.get("expert_traces", []):
                            widths = wave.get("tree_width_by_depth")
                            parents = wave.get("parents")
                            depths = wave.get("depth")
                            if widths is None or parents is None or depths is None:
                                continue

                            iteration = wave.get("iteration", -1)
                            accepted_len = wave.get("accepted_length", 0)

                            parents = _to_list(parents)
                            depths = _to_list(depths)
                            widths = _to_list(widths)
                            max_depth = len(widths) - 1

                            # compute children counts
                            children_counts = [0] * len(parents)
                            for node_idx, parent_idx in enumerate(parents):
                                if parent_idx >= 0:
                                    children_counts[parent_idx] += 1

                            # map depth -> nodes indices
                            depth_to_nodes: Dict[int, List[int]] = {}
                            for node_idx, depth in enumerate(depths):
                                depth_to_nodes.setdefault(depth, []).append(node_idx)

                            # compute per-depth branching stats
                            branch_stats: Dict[int, Dict[str, float]] = {}
                            for depth, nodes in depth_to_nodes.items():
                                branch_counts = [children_counts[idx] for idx in nodes]
                                arr = np.asarray(branch_counts, dtype=float)
                                branch_stats[depth] = {
                                    "branch_mean": float(arr.mean()),
                                    "branch_std": float(arr.std()),
                                    "branch_q25": float(np.percentile(arr, 25)),
                                    "branch_q75": float(np.percentile(arr, 75)),
                                }

                            # total expert cost (sum over layers) if available
                            layer_stats = wave.get("layer_expert_stats") or {}
                            total_unique = 0.0
                            for layer_data in layer_stats.values():
                                total_unique += float(layer_data.get("total_unique", 0))
                            if total_unique == 0 and "total_unique_experts" in wave:
                                total_unique = float(wave.get("total_unique_experts", 0))

                            experts_block = wave.get("experts") or {}
                            layer0_unique = float("nan")
                            if "0" in experts_block:
                                layer0_data = experts_block["0"].get("experts", [])
                                if isinstance(layer0_data, list) and layer0_data and isinstance(layer0_data[0], list):
                                    per_node_layer0 = layer0_data[0]
                                else:
                                    per_node_layer0 = layer0_data
                                union0: set[int] = set()
                                for node_experts in per_node_layer0:
                                    union0.update(int(e) for e in node_experts)
                                layer0_unique = float(len(union0))

                            cycle_rows.append(
                                {
                                    "dataset": label,
                                    "iteration": iteration,
                                    "max_depth": max_depth,
                                    "total_nodes": len(parents),
                                    "accepted_len": accepted_len,
                                    "total_unique": total_unique if total_unique > 0 else np.nan,
                                    "layer0_unique": layer0_unique,
                                }
                            )

                            for depth in range(len(widths)):
                                stats_depth = branch_stats.get(depth, {
                                    "branch_mean": np.nan,
                                    "branch_std": np.nan,
                                    "branch_q25": np.nan,
                                    "branch_q75": np.nan,
                                })
                                per_depth_rows.append(
                                    {
                                        "dataset": label,
                                        "iteration": iteration,
                                        "depth": depth,
                                        "width": widths[depth],
                                        "branch_mean": stats_depth["branch_mean"],
                                        "branch_std": stats_depth["branch_std"],
                                        "branch_q25": stats_depth["branch_q25"],
                                        "branch_q75": stats_depth["branch_q75"],
                                        "total_unique": total_unique if total_unique > 0 else np.nan,
                                        "accepted_len": accepted_len,
                                    }
                                )

    return pd.DataFrame(per_depth_rows), pd.DataFrame(cycle_rows)


def summarize(per_depth_df: pd.DataFrame) -> pd.DataFrame:
    per_depth_df = per_depth_df.copy()
    per_depth_df["width"] = per_depth_df["width"].astype(float)
    summary = per_depth_df.groupby(["dataset", "depth"]).agg(
        width_mean=("width", "mean"),
        width_std=("width", "std"),
        width_q25=("width", lambda x: np.percentile(x, 25)),
        width_q75=("width", lambda x: np.percentile(x, 75)),
        branch_mean_mean=("branch_mean", "mean"),
        branch_mean_std=("branch_mean", "std"),
        branch_q25_mean=("branch_q25", "mean"),
        branch_q75_mean=("branch_q75", "mean"),
        count=("width", "count"),
    ).reset_index()
    return summary


def width_cost_correlation(per_depth_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, depth), group in per_depth_df.groupby(["dataset", "depth"]):
        if group["width"].std() == 0 or group["total_unique"].isna().all():
            corr = np.nan
        else:
            corr = group["width"].corr(group["total_unique"])
        rows.append({"dataset": dataset, "depth": depth, "width_total_unique_corr": corr})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze draft tree structure")
    parser.add_argument("trace_files", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/combined/tree_structure"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.labels or [path.parent.name for path in args.trace_files]
    if len(labels) != len(args.trace_files):
        raise ValueError("Number of labels must match trace files")

    per_depth_df, cycle_df = extract_cycles(args.trace_files, labels)
    if per_depth_df.empty:
        print("No tree data found in traces")
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_depth_path = out_dir / "per_depth_metrics.csv"
    cycle_path = out_dir / "cycle_metrics.csv"
    summary_path = out_dir / "per_depth_summary.csv"
    corr_path = out_dir / "width_cost_correlations.csv"

    per_depth_df.to_csv(per_depth_path, index=False)
    cycle_df.to_csv(cycle_path, index=False)

    summary_df = summarize(per_depth_df)
    summary_df.to_csv(summary_path, index=False)

    corr_df = width_cost_correlation(per_depth_df.dropna(subset=["total_unique"]))
    corr_df.to_csv(corr_path, index=False)

    print("Wrote per-depth metrics to", per_depth_path)
    print("Wrote cycle metrics to", cycle_path)
    print("Per-depth summary:")
    print(summary_df.head())
    print("Width vs total expert correlation:")
    print(corr_df)


if __name__ == "__main__":
    main()
