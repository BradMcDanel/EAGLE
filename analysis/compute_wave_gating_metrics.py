#!/usr/bin/env python3
"""Compute real-unit wave gating metrics across lambda values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from analysis.train_wave_survival_model import WAVE_FEATURES, build_wave_dataframe


def parse_float_list(raw: str | None) -> List[float]:
    if not raw:
        return []
    vals: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    return vals


def build_tables(trace_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_df = pd.read_parquet(trace_path)
    wave_df = build_wave_dataframe(node_df)
    wave_df = wave_df.assign(
        wave_id=np.arange(len(wave_df)),
        wave_cost=wave_df["wave_total_unique"],
        accept_rate=wave_df["accepted_count"] / wave_df["nodes"],
        trace_id=lambda df_: df_["trace_file"].astype(str) + "::" + df_["record_index"].astype(str),
    )
    node_df = node_df.copy()
    node_df["trace_id"] = node_df["trace_file"].astype(str) + "::" + node_df["record_index"].astype(str)
    node_df = node_df.merge(
        wave_df[["trace_file", "record_index", "iteration", "wave_id", "iteration_index"]],
        on=["trace_file", "record_index", "iteration"],
        how="left",
    )
    return node_df, wave_df


def compute_metrics(node_df: pd.DataFrame, wave_df: pd.DataFrame, mask: pd.Series) -> Dict[str, float]:
    wave_ids_kept = set(wave_df.index[mask])

    dedup = (
        node_df.sort_values(["trace_id", "node", "iteration_index"])
        .drop_duplicates(subset=["trace_id", "node"])
        .reset_index(drop=True)
    )

    dedup_mask = dedup["wave_id"].isin(wave_ids_kept)

    total_cost = wave_df["wave_cost"].sum()
    kept_cost = wave_df.loc[mask, "wave_cost"].sum()
    total_accept = dedup["accepted"].sum()
    kept_accept = dedup.loc[dedup_mask, "accepted"].sum()

    trace_ids = wave_df["trace_id"].unique()
    baseline_cost_per_tree = wave_df.groupby("trace_id")["wave_cost"].sum()
    kept_cost_per_tree = (
        wave_df.loc[mask]
        .groupby("trace_id")["wave_cost"]
        .sum()
        .reindex(trace_ids, fill_value=0.0)
    )
    baseline_accept_per_tree = dedup.groupby("trace_id")["accepted"].sum()
    kept_accept_per_tree = dedup.loc[dedup_mask].groupby("trace_id")["accepted"].sum().reindex(trace_ids, fill_value=0.0)

    baseline_mean_unique_per_wave = wave_df["wave_cost"].mean()
    kept_mean_unique_per_wave = wave_df.loc[mask, "wave_cost"].mean() if mask.any() else 0.0
    baseline_mean_accepted_per_wave = wave_df["accepted_count"].mean()
    kept_mean_accepted_per_wave = (
        wave_df.loc[mask, "accepted_count"].mean() if mask.any() else 0.0
    )

    return {
        "cost_kept_frac": kept_cost / total_cost if total_cost else 0.0,
        "accept_kept_frac": kept_accept / total_accept if total_accept else 0.0,
        "mean_cost_per_tree": kept_cost_per_tree.mean(),
        "baseline_mean_cost_per_tree": baseline_cost_per_tree.mean(),
        "mean_unique_experts_per_tree": kept_cost_per_tree.mean(),
        "baseline_mean_unique_experts_per_tree": baseline_cost_per_tree.mean(),
        "mean_accepted_per_tree": kept_accept_per_tree.mean(),
        "baseline_mean_accepted_per_tree": baseline_accept_per_tree.mean(),
        "mean_unique_experts_per_wave": kept_mean_unique_per_wave,
        "baseline_mean_unique_experts_per_wave": baseline_mean_unique_per_wave,
        "mean_accepted_per_wave": kept_mean_accepted_per_wave,
        "baseline_mean_accepted_per_wave": baseline_mean_accepted_per_wave,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--wave-model", type=Path, help="Fitted wave survival model (for predicted mode)")
    parser.add_argument("--lambda", dest="lam", type=float, help="Single lambda value")
    parser.add_argument("--lambda-list", type=str, help="Comma-separated lambda values")
    parser.add_argument("--include-oracle", action="store_true", help="Also compute oracle curve")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lambdas = parse_float_list(args.lambda_list)
    if args.lam is not None:
        lambdas.append(args.lam)
    if not lambdas:
        raise ValueError("Provide --lambda or --lambda-list")

    node_df, wave_df = build_tables(args.trace)

    rows: List[Dict[str, float]] = []

    baseline_mask = pd.Series(True, index=wave_df.index)
    baseline_metrics = compute_metrics(node_df, wave_df, baseline_mask)
    baseline_metrics.update({"lambda": np.nan, "mode": "baseline"})
    rows.append(baseline_metrics)

    if args.wave_model is None:
        raise ValueError("--wave-model required for predicted mode")
    import joblib

    model = joblib.load(args.wave_model)
    preds = model.predict(wave_df[WAVE_FEATURES])
    wave_cost = wave_df["wave_cost"].to_numpy()

    for lam in lambdas:
        util_pred = preds - lam * wave_cost
        mask_pred = util_pred > 0
        metrics_pred = compute_metrics(node_df, wave_df, mask_pred)
        metrics_pred.update({"lambda": lam, "mode": "predicted"})
        rows.append(metrics_pred)

        if args.include_oracle:
            util_oracle = wave_df["accepted_count"].to_numpy() - lam * wave_cost
            mask_oracle = util_oracle > 0
            metrics_oracle = compute_metrics(node_df, wave_df, mask_oracle)
            metrics_oracle.update({"lambda": lam, "mode": "oracle"})
            rows.append(metrics_oracle)

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
