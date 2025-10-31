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

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


TOTAL_LAYER0_EXPERTS = 64.0

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
    "depth_norm",
    "iteration_norm",
    "subtree_fraction",
    "width_fraction",
    "layer0_cumulative_frac",
    "layer0_unique_frac",
    "layer0_remaining_frac",
    "depth_wave_count",
    "depth_wave_frac",
    "parent_branching",
    "iteration_layer0_marginal_sum",
    "iteration_layer0_cumulative_max",
    "iteration_layer0_unique_frac",
    "depth_wave_layer0_marginal_sum",
    "depth_wave_layer0_cumulative_mean",
    "sibling_layer0_marginal_sum",
    "sibling_layer0_marginal_max",
]

PARENT_FEATURES: List[str] = [
    "parent_total_marginal",
    "parent_higher_marginal",
    "parent_layer0_cumulative",
    "parent_depth",
    "parent_layer0_cumulative_frac",
    "parent_layer0_remaining_frac",
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
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("results/combined/cost_model.joblib"),
        help="Optional path to save the trained regressor (set to '' to skip).",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
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

    tree_depth_denom = df["tree_depth"].clip(lower=1)
    tree_size_denom = df["tree_total_nodes"].clip(lower=1)
    df["depth_norm"] = df["depth"] / tree_depth_denom
    if "iteration" in df.columns:
        max_iteration = df.groupby("sample_id")["iteration"].transform("max").replace(0, 1)
        df["iteration_norm"] = df["iteration"] / max_iteration
    else:
        df["iteration_norm"] = 0.0
    df["subtree_fraction"] = df["subtree_size"] / tree_size_denom
    df["width_fraction"] = df["tree_width_at_depth"].fillna(0.0) / tree_size_denom

    df["layer0_cumulative_frac"] = df["layer_0_cumulative"] / TOTAL_LAYER0_EXPERTS
    df["layer0_unique_frac"] = df["layer0_unique"] / TOTAL_LAYER0_EXPERTS
    df["layer0_remaining_frac"] = (
        (TOTAL_LAYER0_EXPERTS - df["layer_0_cumulative"]).clip(lower=0.0) / TOTAL_LAYER0_EXPERTS
    )

    if {"sample_id", "iteration", "node", "depth"}.issubset(df.columns):
        iteration_counts = df.groupby(["sample_id", "iteration"])["node"].transform("count").replace(0, 1)
        depth_wave_counts = (
            df.groupby(["sample_id", "iteration", "depth"])["node"].transform("count").astype(float)
        )
        df["depth_wave_count"] = depth_wave_counts
        df["depth_wave_frac"] = depth_wave_counts / iteration_counts
    else:
        df["depth_wave_count"] = 0.0
        df["depth_wave_frac"] = 0.0

    if {"sample_id", "iteration", "parent"}.issubset(df.columns):
        parent_branching = (
            df.groupby(["sample_id", "iteration", "parent"])["node"].transform("count").astype(float)
        )
        df["parent_branching"] = parent_branching
    else:
        df["parent_branching"] = 0.0

    if {"sample_id", "iteration"}.issubset(df.columns):
        iter_group = df.groupby(["sample_id", "iteration"])
        df["iteration_layer0_marginal_sum"] = iter_group["layer_0_marginal"].transform("sum")
        df["iteration_layer0_cumulative_max"] = iter_group["layer_0_cumulative"].transform("max")
        df["iteration_layer0_unique_frac"] = df["iteration_layer0_cumulative_max"] / TOTAL_LAYER0_EXPERTS
    else:
        df["iteration_layer0_marginal_sum"] = 0.0
        df["iteration_layer0_cumulative_max"] = 0.0
        df["iteration_layer0_unique_frac"] = 0.0

    if {"sample_id", "iteration", "depth"}.issubset(df.columns):
        depth_iter_group = df.groupby(["sample_id", "iteration", "depth"])
        df["depth_wave_layer0_marginal_sum"] = depth_iter_group["layer_0_marginal"].transform("sum")
        df["depth_wave_layer0_cumulative_mean"] = depth_iter_group["layer_0_cumulative"].transform("mean")
    else:
        df["depth_wave_layer0_marginal_sum"] = 0.0
        df["depth_wave_layer0_cumulative_mean"] = 0.0

    if {"sample_id", "iteration", "parent"}.issubset(df.columns):
        sibling_group = df.groupby(["sample_id", "iteration", "parent"])
        df["sibling_layer0_marginal_sum"] = sibling_group["layer_0_marginal"].transform("sum")
        df["sibling_layer0_marginal_max"] = sibling_group["layer_0_marginal"].transform("max")
    else:
        df["sibling_layer0_marginal_sum"] = 0.0
        df["sibling_layer0_marginal_max"] = 0.0

    for column in [
        "depth_norm",
        "iteration_norm",
        "subtree_fraction",
        "width_fraction",
        "layer0_cumulative_frac",
        "layer0_unique_frac",
        "layer0_remaining_frac",
        "depth_wave_count",
        "depth_wave_frac",
        "parent_branching",
        "iteration_layer0_marginal_sum",
        "iteration_layer0_cumulative_max",
        "iteration_layer0_unique_frac",
        "depth_wave_layer0_marginal_sum",
        "depth_wave_layer0_cumulative_mean",
        "sibling_layer0_marginal_sum",
        "sibling_layer0_marginal_max",
    ]:
        df[column] = df[column].fillna(0.0)
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
    merged["parent_layer0_cumulative_frac"] = merged["parent_layer0_cumulative"] / TOTAL_LAYER0_EXPERTS
    merged["parent_layer0_remaining_frac"] = (
        (TOTAL_LAYER0_EXPERTS - merged["parent_layer0_cumulative"]).clip(lower=0.0) / TOTAL_LAYER0_EXPERTS
    )

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

    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_out)

    if args.predictions_out:
        preds = model.predict(X_test)
        dump_predictions(test_df, preds, args.predictions_out)

    print("Higher-layer cost model metrics (hold-out):")
    for key, value in metrics.items():
        print(f"  {key:>10s}: {value:.4f}")

    baseline_mae = float(np.mean(np.abs(y_test - y_test.mean())))
    print(f"\nBaseline (predict mean) MAE: {baseline_mae:.4f}")
    print(f"Training samples: {len(X_train)}, hold-out samples: {len(X_test)}")

    try:
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            random_state=args.seed,
            n_jobs=1,
        )
        scores = perm.importances_mean
        order = np.argsort(scores)[::-1]
        print("\nTop features (permutation importance):")
        for idx in order[: min(15, len(order))]:
            print(f"  {ALL_FEATURES[idx]:>25s}: {scores[idx]:.6f}")
    except Exception as exc:
        print(f"\nWarning: permutation importance failed ({exc})")


if __name__ == "__main__":
    main()
