#!/usr/bin/env python3
"""Budgeted pruning simulation using layer-0 expert unions and node utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd


def parse_layer0_set(value: str) -> Set[int]:
    if isinstance(value, str) and value:
        return set(map(int, value.split()))
    return set()


def simulate_cycle(group: pd.DataFrame, budget: float) -> Dict[str, float]:
    # Sort by depth ascending, then by log_prob descending to maintain tree structure
    group = group.copy()
    group["log_prob"].fillna(-1e9, inplace=True)
    group.sort_values(["depth", "log_prob"], ascending=[True, False], inplace=True)

    kept_nodes: Set[int] = set()
    tree_cost: Set[int] = set()
    baseline_cost: Set[int] = set()

    layer0_sets = [parse_layer0_set(val) for val in group["layer0_set"]]

    for node_set in layer0_sets:
        baseline_cost.update(node_set)

    parent_map = dict(zip(group["node"], group["parent"]))
    node_to_sets = dict(zip(group["node"], layer0_sets))

    for _, row in group.iterrows():
        node = int(row["node"])
        parent = int(row["parent"])
        if parent >= 0 and parent not in kept_nodes:
            continue

        experts = node_to_sets[node]
        new_experts = experts - tree_cost
        if row["depth"] == 0:
            # root always kept; cost must be within budget
            tree_cost.update(experts)
            kept_nodes.add(node)
            continue

        if len(tree_cost) + len(new_experts) <= budget:
            kept_nodes.add(node)
            tree_cost.update(experts)

    accepted_nodes = group[group["accepted"] == 1]["node"].astype(int).tolist()
    accepted_kept = sum(1 for node in accepted_nodes if node in kept_nodes)

    return {
        "baseline_cost": float(len(baseline_cost)),
        "kept_cost": float(len(tree_cost)),
        "baseline_nodes": float(len(group)),
        "kept_nodes": float(len(kept_nodes)),
        "baseline_accept": float(len(accepted_nodes)),
        "kept_accept": float(accepted_kept),
    }


def simulate(df: pd.DataFrame, budgets: Iterable[float]) -> pd.DataFrame:
    results = []
    grouped = df.groupby(["dataset", "iteration"])
    for budget in budgets:
        total = {
            "baseline_cost": 0.0,
            "kept_cost": 0.0,
            "baseline_nodes": 0.0,
            "kept_nodes": 0.0,
            "baseline_accept": 0.0,
            "kept_accept": 0.0,
        }
        for _, group in grouped:
            stats = simulate_cycle(group, budget)
            for key in total:
                total[key] += stats[key]

        results.append(
            {
                "budget": budget,
                "nodes_reduction_pct": 1 - total["kept_nodes"] / total["baseline_nodes"],
                "experts_reduction_pct": 1 - total["kept_cost"] / total["baseline_cost"],
                "acceptance_drop_pct": 1 - total["kept_accept"] / total["baseline_accept"],
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate budgeted pruning using layer-0 unions")
    parser.add_argument("node_csv", type=Path)
    parser.add_argument("--budgets", nargs="*", type=float, default=[30, 35, 40, 45, 50, 55, 60])
    parser.add_argument("--out", type=Path, default=Path("results/combined/budget_simulation.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.node_csv)
    required = {"dataset", "iteration", "node", "parent", "depth", "layer0_set", "log_prob", "accepted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in node data: {missing}")

    result = simulate(df, args.budgets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result)


if __name__ == "__main__":
    main()

