#!/usr/bin/env python3
"""Compute expert-cost statistics with bootstrap confidence intervals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def bootstrap_ci(
    values: np.ndarray,
    num_samples: int,
    confidence: float,
    random_state: int | None,
) -> Tuple[float, float]:
    """Return non-parametric bootstrap confidence interval for the mean."""

    rng = np.random.default_rng(random_state)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")

    stats = []
    for _ in range(num_samples):
        sample = rng.choice(values, size=n, replace=True)
        stats.append(sample.mean())
    lower = np.percentile(stats, (1 - confidence) / 2 * 100)
    upper = np.percentile(stats, (1 + confidence) / 2 * 100)
    return float(lower), float(upper)


def compute_metrics(
    df: pd.DataFrame,
    group_cols: Iterable[str],
    num_bootstrap: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {col: key for col, key in zip(group_cols, keys)}

        values = group["experts_per_token"].to_numpy()
        values_per_layer = group["experts_per_token_per_layer"].to_numpy()
        weights = group["total_weight_per_token"].to_numpy()

        lower, upper = bootstrap_ci(
            values[~np.isnan(values)], num_samples=num_bootstrap, confidence=confidence, random_state=seed
        )
        per_layer_lower, per_layer_upper = bootstrap_ci(
            values_per_layer[~np.isnan(values_per_layer)],
            num_samples=num_bootstrap,
            confidence=confidence,
            random_state=seed,
        )
        w_lower, w_upper = bootstrap_ci(
            weights[~np.isnan(weights)], num_samples=num_bootstrap, confidence=confidence, random_state=seed
        )

        row = {
            **base,
            "count": len(group),
            "experts_per_token_mean": float(np.nanmean(values)),
            "experts_per_token_median": float(np.nanmedian(values)),
            "experts_per_token_std": float(np.nanstd(values)),
            "experts_per_token_q25": float(np.nanpercentile(values, 25)),
            "experts_per_token_q75": float(np.nanpercentile(values, 75)),
            "experts_per_token_ci_lo": lower,
            "experts_per_token_ci_hi": upper,
            "experts_per_token_per_layer_mean": float(np.nanmean(values_per_layer)),
            "experts_per_token_per_layer_median": float(np.nanmedian(values_per_layer)),
            "experts_per_token_per_layer_std": float(np.nanstd(values_per_layer)),
            "experts_per_token_per_layer_ci_lo": per_layer_lower,
            "experts_per_token_per_layer_ci_hi": per_layer_upper,
            "total_weight_per_token_mean": float(np.nanmean(weights)),
            "total_weight_per_token_median": float(np.nanmedian(weights)),
            "total_weight_per_token_ci_lo": w_lower,
            "total_weight_per_token_ci_hi": w_upper,
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize expert cost statistics")
    parser.add_argument("waves", type=Path, help="CSV file produced by analyze_tree_experts")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path for summary statistics")
    parser.add_argument(
        "--num-bootstrap", type=int, default=1000, help="Number of bootstrap samples"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95, help="Bootstrap confidence level"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for bootstrap resampling"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.waves)

    denom = df["accepted_len"].clip(lower=1)
    df["experts_per_token"] = df["accepted_unique_total"] / denom
    if "layer_count" in df.columns:
        layer_counts = df["layer_count"].replace(0, np.nan)
    else:
        layer_columns = [col for col in df.columns if col.startswith("layer_") and col.endswith("_accepted_unique")]
        num_layers = len(layer_columns)
        if num_layers == 0:
            raise ValueError("No layer information available to normalize per layer")
        layer_counts = num_layers
    df["experts_per_token_per_layer"] = df["experts_per_token"] / layer_counts

    per_layer_weight_cols = [col for col in df.columns if col.endswith("_accepted_weight")]
    if per_layer_weight_cols:
        df["accepted_weight_total"] = df[per_layer_weight_cols].sum(axis=1)
        df["total_weight_per_token"] = df["accepted_weight_total"] / denom
    else:
        df["total_weight_per_token"] = np.nan

    dataset_depth_summary = compute_metrics(
        df,
        group_cols=["dataset", "depth"],
        num_bootstrap=args.num_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
    )

    dataset_summary = compute_metrics(
        df,
        group_cols=["dataset"],
        num_bootstrap=args.num_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
    )

    summary = {
        "dataset_depth": dataset_depth_summary,
        "dataset": dataset_summary,
    }

    if args.out:
        out_path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_depth_summary.to_csv(out_path.with_name(out_path.stem + "_by_depth.csv"), index=False)
        dataset_summary.to_csv(out_path.with_name(out_path.stem + "_overall.csv"), index=False)

    for name, table in summary.items():
        print("==", name, "==")
        print(table)


if __name__ == "__main__":
    main()
