#!/usr/bin/env python3
"""Train a cost regressor that estimates higher-layer MoE usage per node.

We predict the additional (non-layer-0) expert cost each draft-tree node
incurs, using only information that is available immediately after
layer-0 routing completes.  This gives a forward-looking estimate of how
expensive it will be to keep expanding a branch before the target model
actually runs all MoE layers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_FEATURES: List[str] = [
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

ALL_FEATURES = BASE_FEATURES + PARENT_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("results/combined/node_layer_marginals.csv"),
        help="Path to node-level CSV produced by viz/node_layer_marginals.py",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of samples reserved for evaluation (sample-level split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples (draft trees) used for training.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("results/combined/cost_model_metrics.json"),
        help="Path to JSON file for evaluation metrics.",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("results/combined/cost_model_holdout.csv"),
        help="CSV for hold-out predictions (set to '' to skip).",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"total_marginal", "layer_0_marginal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def build_cycle_id(df: pd.DataFrame) -> pd.Series:
    keys = [
        "dataset",
        "trace_file",
        "record_index",
        "choice_index",
        "stats_index",
        "wave_index",
        "iteration",
    ]
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns needed for cycle grouping: {missing}")
    return df[keys].astype(str).agg("::".join, axis=1)


def count_layer0_experts(value) -> int:
    if isinstance(value, str) and value:
        return len(value.split())
    return 0


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["depth"] > 0]  # root provides no incremental signal

    required = {
        "routing_weight",
        "subtree_weight",
        "layer_0_marginal",
        "layer_0_cumulative",
        "draft_log_prob",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    df = df.dropna(subset=list(required))

    df["layer0_unique"] = df.get("layer0_set", "").apply(count_layer0_experts)
    df["higher_marginal"] = df["total_marginal"] - df["layer_0_marginal"]
    df["cycle_id"] = build_cycle_id(df)
    df["sample_id"] = df["trace_file"].astype(str) + "::" + df["record_index"].astype(str)
    return df


def add_parent_features(df: pd.DataFrame) -> pd.DataFrame:
    parent_cols = [
        "cycle_id",
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

    merged = df.merge(
        parent_df,
        how="left",
        on=["cycle_id", "parent"],
    )

    # Fill NaNs for root children (parent == -1)
    for col in ["parent_total_marginal", "parent_higher_marginal", "parent_layer0_cumulative"]:
        merged[col] = merged[col].fillna(0.0)
    merged["parent_depth"] = merged["parent_depth"].fillna(0.0)

    return merged


def subsample_samples(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples is None:
        return df
    sample_ids = df["sample_id"].unique()
    if len(sample_ids) <= max_samples:
        return df
    rng = np.random.default_rng(seed)
    keep_ids = rng.choice(sample_ids, size=max_samples, replace=False)
    return df[df["sample_id"].isin(keep_ids)]


def split_samples(
    df: pd.DataFrame,
    test_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_ids = df["sample_id"].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_frac, random_state=seed)
    train_mask = df["sample_id"].isin(train_ids)
    return df[train_mask], df[~train_mask]


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_regressor(
    model: HistGradientBoostingRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    preds = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "y_test_mean": float(np.mean(y_test)),
    }


def write_metrics(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(metrics, fh, indent=2)


def dump_predictions(
    df_test: pd.DataFrame,
    preds: np.ndarray,
    path: Path,
) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    out_cols = [
        "dataset",
        "trace_file",
        "record_index",
        "choice_index",
        "stats_index",
        "wave_index",
        "iteration",
        "node",
        "parent",
        "depth",
        "total_marginal",
        "higher_marginal",
        "sample_id",
    ]
    available_cols = [c for c in out_cols if c in df_test.columns]
    out_df = df_test[available_cols].copy()
    out_df["predicted_higher_marginal"] = preds
    out_df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    df = load_dataset(args.data)
    df = preprocess(df)
    df = add_parent_features(df)
    df = subsample_samples(df, args.max_samples, args.seed)

    target = df["higher_marginal"]

    train_df, test_df = split_samples(df, args.test_frac, args.seed)
    X_train = train_df[ALL_FEATURES]
    y_train = train_df["higher_marginal"]

    X_test = test_df[ALL_FEATURES]
    y_test = test_df["higher_marginal"]

    model = train_regressor(X_train, y_train, args.seed)
    metrics = evaluate_regressor(model, X_test, y_test)
    write_metrics(metrics, args.metrics_out)

    if args.predictions_out:
        preds = model.predict(X_test)
        dump_predictions(test_df, preds, args.predictions_out)

    print("Higher-layer cost model metrics (hold-out):")
    for key, value in metrics.items():
        print(f"  {key:>10s}: {value:.4f}")

    baseline_mae = float(np.mean(np.abs(y_test - y_test.mean())))
    print(f"\nBaseline (predict mean) MAE: {baseline_mae:.4f}")
    print(f"Training samples: {len(X_train)}, hold-out samples: {len(X_test)}")


if __name__ == "__main__":
    main()
