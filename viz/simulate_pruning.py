#!/usr/bin/env python3
"""Simulate pruning draft trees using precomputed node-layer marginals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def load_node_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dataset", "iteration", "node", "parent", "depth", "layer_0_marginal", "total_marginal", "accepted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in node data: {missing}")
    return df


def propagate_pruning(df: pd.DataFrame, threshold: float) -> pd.Series:
    pruned = pd.Series(False, index=df.index)
    to_prune_nodes = df.loc[(df["layer_0_marginal"] > threshold) & (df["depth"] > 0), "node"].tolist()
    if not to_prune_nodes:
        return pruned

    children = df.groupby("parent")["node"].apply(list).to_dict()
    node_to_idx = {row.node: idx for idx, row in df.iterrows()}

    queue = list(to_prune_nodes)
    while queue:
        node = queue.pop()
        idx = node_to_idx.get(node)
        if idx is None or pruned.iat[idx]:
            continue
        pruned.iat[idx] = True
        queue.extend(children.get(node, []))

    return pruned


def simulate(df: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    results: List[Dict] = []
    for thresh in thresholds:
        pruned_mask = pd.Series(False, index=df.index)
        for (dataset, iteration), group in df.groupby(["dataset", "iteration"]):
            local = group.sort_values("depth").reset_index()
            pruned_local = propagate_pruning(local, thresh)
            pruned_mask.loc[local["index"]] = pruned_local.values

        nodes_total = len(df)
        nodes_pruned = pruned_mask.sum()

        experts_total = df["total_marginal"].sum()
        experts_pruned = df.loc[pruned_mask, "total_marginal"].sum()

        accepted_total = df["accepted"].sum()
        accepted_pruned = df.loc[pruned_mask, "accepted"].sum()

        results.append(
            {
                "threshold": thresh,
                "nodes_total": nodes_total,
                "nodes_pruned": nodes_pruned,
                "experts_total": experts_total,
                "experts_pruned": experts_pruned,
                "accepted_total": accepted_total,
                "accepted_pruned": accepted_pruned,
            }
        )

    result_df = pd.DataFrame(results)
    result_df["nodes_reduction_pct"] = result_df["nodes_pruned"] / result_df["nodes_total"]
    result_df["experts_reduction_pct"] = result_df["experts_pruned"] / result_df["experts_total"]
    result_df["acceptance_drop_pct"] = result_df["accepted_pruned"] / result_df["accepted_total"]
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate pruning using layer-0 marginal thresholds")
    parser.add_argument("node_csv", type=Path)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[3, 4, 5, 6, 7])
    parser.add_argument("--out", type=Path, default=Path("results/combined/pruning_simulation.csv"))
    args = parser.parse_args()

    df = load_node_data(args.node_csv)
    result = simulate(df, args.thresholds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result)


if __name__ == "__main__":
    main()
