#!/usr/bin/env python3
"""Train a wave-level regressor that predicts surviving nodes per draft iteration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


WAVE_FEATURES: List[str] = [
    "nodes",
    "wave_total_unique",
    "wave_layer0_unique",
    "wave_higher_unique",
    "unique_total_per_node",
    "unique_layer0_per_node",
    "unique_higher_per_node",
    "routing_weight_sum",
    "routing_weight_mean",
    "subtree_weight_sum",
    "subtree_weight_mean",
    "depth_mean",
    "depth_std",
    "frontier_width_mean",
    "frontier_width_std",
    "unique_experts_layer0",
    # Cumulative context
    "cumulative_wave_total_unique",
    "cumulative_wave_higher_unique",
    "iteration_index",
    "tree_depth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Trace table produced with --trace-schema analysis (CSV or Parquet)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("results/combined/wave_survival_model.joblib"),
        help="Path to save the fitted model (set to '' to skip).",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("results/combined/wave_survival_metrics.json"),
        help="Where to write regression metrics (JSON).",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("results/combined/wave_survival_holdout.csv"),
        help="Where to dump hold-out predictions (CSV, set to '' to skip).",
    )
    parser.add_argument("--test-frac", type=float, default=0.2, help="Hold-out fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_wave_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "trace_file",
        "record_index",
        "iteration",
        "depth",
        "accepted",
        "total_marginal",
        "higher_marginal",
        "layer_0_marginal",
        "wave_total_unique",
        "wave_layer0_unique",
        "wave_higher_unique",
        "routing_weight",
        "subtree_weight",
        "tree_width_at_depth",
        "tree_depth",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df.copy()
    df["accepted"] = df["accepted"].fillna(0).astype(int)

    group_keys = ["trace_file", "record_index", "iteration"]
    wave_df = (
        df.groupby(group_keys)
        .agg(
            nodes=("node", "count"),
            accepted_count=("accepted", "sum"),
            routing_weight_sum=("routing_weight", "sum"),
            routing_weight_mean=("routing_weight", "mean"),
            subtree_weight_sum=("subtree_weight", "sum"),
            subtree_weight_mean=("subtree_weight", "mean"),
            depth_mean=("depth", "mean"),
            depth_std=("depth", "std"),
            frontier_width_mean=("tree_width_at_depth", "mean"),
            frontier_width_std=("tree_width_at_depth", "std"),
            wave_total_unique=("wave_total_unique", "max"),
            wave_accepted_unique=("wave_accepted_unique", "max"),
            wave_layer0_unique=("wave_layer0_unique", "max"),
            wave_higher_unique=("wave_higher_unique", "max"),
            tree_depth=("tree_depth", "max"),
        )
        .reset_index()
    )

    wave_df["depth_std"] = wave_df["depth_std"].fillna(0.0)
    wave_df["frontier_width_std"] = wave_df["frontier_width_std"].fillna(0.0)
    for col in ["wave_total_unique", "wave_accepted_unique", "wave_layer0_unique", "wave_higher_unique"]:
        wave_df[col] = wave_df[col].fillna(0.0)

    layer0_unique = (
        df.loc[:, group_keys + ["layer0_set"]]
        .drop_duplicates()
        .groupby(group_keys)["layer0_set"]
        .apply(lambda vals: len(set(e for v in vals if isinstance(v, str) for e in v.split())))
        .reset_index(name="unique_experts_layer0")
    )
    wave_df = wave_df.merge(layer0_unique, on=group_keys, how="left")
    wave_df["unique_experts_layer0"] = wave_df["unique_experts_layer0"].fillna(0)

    wave_df = wave_df.sort_values(group_keys).reset_index(drop=True)

    wave_df["iteration_index"] = wave_df.groupby(["trace_file", "record_index"]).cumcount()

    wave_df["unique_total_per_node"] = wave_df["wave_total_unique"] / wave_df["nodes"].clip(lower=1)
    wave_df["unique_layer0_per_node"] = wave_df["wave_layer0_unique"] / wave_df["nodes"].clip(lower=1)
    wave_df["unique_higher_per_node"] = wave_df["wave_higher_unique"] / wave_df["nodes"].clip(lower=1)

    wave_df["cumulative_wave_total_unique"] = wave_df.groupby(["trace_file", "record_index"])[
        "wave_total_unique"
    ].cumsum()
    wave_df["cumulative_wave_higher_unique"] = wave_df.groupby(["trace_file", "record_index"])[
        "wave_higher_unique"
    ].cumsum()

    return wave_df


def split_train_test(
    df: pd.DataFrame, test_frac: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = df[["trace_file", "record_index"]].drop_duplicates()
    train_keys, test_keys = train_test_split(
        keys, test_size=test_frac, random_state=seed, shuffle=True
    )
    train_mask = df.set_index(["trace_file", "record_index"]).index.isin(
        train_keys.set_index(["trace_file", "record_index"]).index
    )
    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[~train_mask].reset_index(drop=True)
    return train_df, test_df


def train_regressor(X: pd.DataFrame, y: pd.Series, seed: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.1, random_state=seed)
    model.fit(X, y)
    return model


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    preds = model.predict(X)
    return {
        "r2": float(r2_score(y, preds)),
        "mae": float(mean_absolute_error(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "y_true_mean": float(np.mean(y)),
        "y_pred_mean": float(np.mean(preds)),
    }


def main() -> None:
    args = parse_args()
    df = load_data(args.data)
    wave_df = build_wave_dataframe(df)

    train_df, test_df = split_train_test(wave_df, args.test_frac, args.seed)
    X_train = train_df[WAVE_FEATURES]
    y_train = train_df["accepted_count"]
    X_test = test_df[WAVE_FEATURES]
    y_test = test_df["accepted_count"]

    model = train_regressor(X_train, y_train, args.seed)
    metrics = evaluate(model, X_test, y_test)

    print("Wave survival model metrics (hold-out):")
    for key, value in metrics.items():
        print(f"  {key:>10s}: {value:.4f}")

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w") as fh:
            json.dump(metrics, fh, indent=2)

    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_out)

    if args.predictions_out:
        preds = model.predict(X_test)
        out_df = test_df[["trace_file", "record_index", "iteration", "nodes", "accepted_count"]].copy()
        out_df["predicted_accepted_count"] = preds
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.predictions_out, index=False)


if __name__ == "__main__":
    main()
