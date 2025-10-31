#!/usr/bin/env python3
"""Evaluate utility-based pruning metrics on precomputed node tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

ACCEPT_FEATURES: List[str] = [
    "depth",
    "children_count",
    "is_leaf",
    "subtree_size",
    "tree_total_nodes",
    "tree_depth",
    "tree_width_at_depth",
    "draft_log_prob",
    "routing_weight",
    "subtree_weight",
    "layer0_unique",
    "layer_0_marginal",
    "layer_0_cumulative",
]

PARENT_FEATURES: List[str] = [
    "parent_total_marginal",
    "parent_higher_marginal",
    "parent_layer0_cumulative",
    "parent_depth",
]

COST_FEATURES: List[str] = ACCEPT_FEATURES + PARENT_FEATURES


def parse_lambdas(raw: Iterable[str]) -> List[float]:
    values: List[float] = []
    for item in raw:
        if "," in item:
            values.extend(parse_lambdas(item.split(",")))
        else:
            values.append(float(item))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="CSV from viz/node_layer_marginals.py")
    parser.add_argument("--accept-model", type=Path, required=True, help="Trained acceptance Joblib model")
    parser.add_argument("--cost-model", type=Path, required=True, help="Trained cost Joblib model")
    parser.add_argument(
        "--lambdas",
        nargs="+",
        default=["0", "1e-5", "5e-5", "1e-4", "5e-4", "1e-3"],
        help="Lambda values to sweep (space or comma separated)",
    )
    parser.add_argument("--out", type=Path, default=Path("results/combined/utility_sweep.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if "log_prob" not in df.columns and "draft_log_prob" in df.columns:
        df["log_prob"] = df["draft_log_prob"]
    if "layer0_unique" not in df.columns and "layer0_set" in df.columns:
        def _count_layer0(value):
            if isinstance(value, str) and value:
                return len(value.split())
            return 0
        df["layer0_unique"] = df["layer0_set"].apply(_count_layer0)

    if any(col not in df.columns for col in PARENT_FEATURES):
        parent_cols = [
            "dataset",
            "trace_file",
            "record_index",
            "choice_index",
            "stats_index",
            "wave_index",
            "iteration",
            "node",
            "total_marginal",
            "higher_marginal",
            "layer_0_cumulative",
            "depth",
        ]
        parent_df = df[parent_cols].rename(
            columns={
                "node": "parent",
                "total_marginal": "parent_total_marginal",
                "higher_marginal": "parent_higher_marginal",
                "layer_0_cumulative": "parent_layer0_cumulative",
                "depth": "parent_depth",
            }
        )
        df = df.merge(
            parent_df,
            how="left",
            on=[
                "dataset",
                "trace_file",
                "record_index",
                "choice_index",
                "stats_index",
                "wave_index",
                "iteration",
                "parent",
            ],
        )
        for col in PARENT_FEATURES:
            df[col] = df[col].fillna(0.0)


    accept_model = joblib.load(args.accept_model)
    cost_model = joblib.load(args.cost_model)

    df["p_accept"] = accept_model.predict_proba(df[ACCEPT_FEATURES])[:, 1]
    df["higher_marginal_hat"] = np.clip(cost_model.predict(df[COST_FEATURES]), a_min=0.0, a_max=None)

    lambdas = parse_lambdas(args.lambdas)
    results: List[dict] = []
    accepted_total = df["accepted"].sum()
    higher_total = df["higher_marginal"].sum()

    groups = {iteration: group.sort_values("depth") for iteration, group in df.groupby("iteration")}

    for lam in lambdas:
        utility_values = df["p_accept"] - lam * df["higher_marginal_hat"]
        keep_mask = pd.Series(False, index=df.index)

        for iteration, group in groups.items():
            keep_local = {}
            for idx, row in group.iterrows():
                if row["depth"] == 0:
                    keep = True
                else:
                    parent_keep = keep_local.get(row["parent"], False)
                    keep = parent_keep and (utility_values.at[idx] >= 0.0)
                keep_local[row["node"]] = keep
                keep_mask.at[idx] = keep

        nodes_kept_pct = keep_mask.mean()
        accepted_pruned = df.loc[~keep_mask, "accepted"].sum()
        higher_pruned = df.loc[~keep_mask, "higher_marginal"].sum()

        results.append(
            {
                "lambda": lam,
                "nodes_kept_pct": nodes_kept_pct,
                "accepted_drop_pct": (accepted_pruned / accepted_total) if accepted_total else 0.0,
                "higher_reduction_pct": (higher_pruned / higher_total) if higher_total else 0.0,
            }
        )

    result_df = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.out, index=False)
    print(result_df)


if __name__ == "__main__":
    main()
