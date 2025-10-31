#!/usr/bin/env python3
"""Evaluate cost/acceptance gating trade-offs on saved training traces."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

# Ensure project root on sys.path when run as a script
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from analysis.train_accept_model import LAYER0_FEATURES
from analysis.train_cost_model import (
    ALL_FEATURES,
    add_parent_features,
    load_dataset,
    preprocess,
)


def parse_float_list(raw: str | None, default: Iterable[float]) -> List[float]:
    if raw is None:
        return list(default)
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        return list(default)
    return values


def compute_stats(
    mask: np.ndarray,
    accepted: np.ndarray,
    actual_cost: np.ndarray,
    total_marginal: np.ndarray,
    total_cost: float,
    accepted_cost: float,
    rejected_cost: float,
) -> dict[str, float]:
    pruned_nodes = mask.sum()
    pruned_cost = float(actual_cost[mask].sum())
    pruned_accepts = int(accepted[mask].sum())
    pruned_reject_cost = float(actual_cost[mask & (accepted == 0)].sum())
    pruned_total_marginal_mean = float(total_marginal[mask].mean()) if pruned_nodes else 0.0

    kept_mask = ~mask
    kept_nodes = kept_mask.sum()
    kept_cost = float(actual_cost[kept_mask].sum())
    kept_total_marginal_mean = float(total_marginal[kept_mask].mean()) if kept_nodes else 0.0
    kept_accepts = int(accepted[kept_mask].sum())
    kept_accept_rate = kept_accepts / kept_nodes if kept_nodes else 0.0

    accepted_total = accepted.sum()
    accepted_pruned_frac = pruned_accepts / accepted_total if accepted_total else 0.0
    kept_accept_frac = 1.0 - accepted_pruned_frac

    stats = {
        "nodes_pruned": float(pruned_nodes),
        "nodes_pruned_frac": float(pruned_nodes) / len(mask) if len(mask) else 0.0,
        "cost_pruned": pruned_cost,
        "cost_pruned_frac": pruned_cost / total_cost if total_cost else 0.0,
        "accepted_pruned": float(pruned_accepts),
        "accepted_pruned_frac": accepted_pruned_frac,
        "kept_accept_frac": kept_accept_frac,
        "accepted_share_within_pruned": pruned_accepts / pruned_nodes if pruned_nodes else 0.0,
        "rejected_cost_pruned_frac": pruned_reject_cost / rejected_cost if rejected_cost else 0.0,
        "pruned_total_marginal_mean": pruned_total_marginal_mean,
        "kept_nodes": float(kept_nodes),
        "kept_cost": kept_cost,
        "kept_cost_frac": kept_cost / total_cost if total_cost else 0.0,
        "kept_total_marginal_mean": kept_total_marginal_mean,
        "kept_accept_rate": kept_accept_rate,
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Training trace (CSV or Parquet)")
    parser.add_argument("--accept-model", type=Path, required=True, help="Joblib acceptance model")
    parser.add_argument("--cost-model", type=Path, required=True, help="Joblib cost model")
    parser.add_argument(
        "--accept-thresholds",
        type=str,
        default=None,
        help="Comma-separated acceptance probability thresholds (prune when prob < τ).",
    )
    parser.add_argument(
        "--cost-quantiles",
        type=str,
        default=None,
        help="Comma-separated cost quantiles (prune when predicted cost ≥ q-th quantile).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/combined/gating_tradeoff.csv"),
        help="Where to write the sweep summary CSV.",
    )
    args = parser.parse_args()

    accept_thresholds = parse_float_list(args.accept_thresholds, default=[0.05, 0.1, 0.15, 0.2])
    cost_quantiles = parse_float_list(args.cost_quantiles, default=[0.7, 0.8, 0.9, 0.95])

    print("Loading dataset...")
    raw_df = load_dataset(args.data)
    proc_df = add_parent_features(preprocess(raw_df))

    for col in set(LAYER0_FEATURES + ALL_FEATURES + ["accepted", "higher_marginal"]):
        if col not in proc_df.columns:
            raise KeyError(f"Required column '{col}' missing after preprocessing.")

    accept_model = joblib.load(args.accept_model)
    cost_model = joblib.load(args.cost_model)

    print("Computing predictions...")
    accept_probs = accept_model.predict_proba(proc_df[LAYER0_FEATURES])[:, 1]
    cost_preds = cost_model.predict(proc_df[ALL_FEATURES])

    accepted = proc_df["accepted"].to_numpy(dtype=np.int32)
    actual_cost = proc_df["higher_marginal"].to_numpy(dtype=np.float64)
    total_marginal = proc_df["total_marginal"].to_numpy(dtype=np.float64)

    total_cost = float(actual_cost.sum())
    accepted_cost = float(actual_cost[accepted == 1].sum())
    rejected_cost = total_cost - accepted_cost

    print(f"Total nodes: {len(proc_df):,}")
    print(f"Total cost: {total_cost:,.0f}")
    print(f"Accepted-cost share: {accepted_cost / total_cost:.2%}")

    rows = []
    baseline_stats = compute_stats(
        mask=np.zeros_like(accepted, dtype=bool),
        accepted=accepted,
        actual_cost=actual_cost,
        total_marginal=total_marginal,
        total_cost=total_cost,
        accepted_cost=accepted_cost,
        rejected_cost=rejected_cost,
    )
    rows.append(
        {
            "accept_threshold": np.nan,
            "cost_quantile": np.nan,
            "cost_threshold": np.nan,
            **baseline_stats,
        }
    )
    for acc_thresh in accept_thresholds:
        for q in cost_quantiles:
            if not 0.0 < q < 1.0:
                raise ValueError(f"Cost quantile must be in (0, 1), got {q}")
            cost_thresh = float(np.quantile(cost_preds, q))
            mask = (accept_probs < acc_thresh) & (cost_preds >= cost_thresh)
            stats = compute_stats(
                mask,
                accepted,
                actual_cost,
                total_marginal,
                total_cost,
                accepted_cost,
                rejected_cost,
            )
            rows.append(
                {
                    "accept_threshold": acc_thresh,
                    "cost_quantile": q,
                    "cost_threshold": cost_thresh,
                    **stats,
                }
            )

    result_df = pd.DataFrame(rows).sort_values(
        ["accept_threshold", "cost_quantile"], ignore_index=True
    )
    baseline_row = result_df[result_df["accept_threshold"].isna()].iloc[0]
    baseline_experts = baseline_row["kept_total_marginal_mean"]
    baseline_accept = baseline_row["kept_accept_frac"]
    result_df["kept_expert_ratio"] = (
        result_df["kept_total_marginal_mean"] / baseline_experts if baseline_experts else 0.0
    )
    result_df["kept_accept_ratio"] = (
        result_df["kept_accept_frac"] / baseline_accept if baseline_accept else 0.0
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.out, index=False)

    print(f"\nSaved sweep summary to {args.out}")
    print(result_df.head())


if __name__ == "__main__":
    main()
