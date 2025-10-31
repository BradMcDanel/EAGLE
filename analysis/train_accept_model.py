#!/usr/bin/env python3
"""Train an acceptance-probability model using layer-0 routing signals.

This script reads the aggregated node-level dataset produced by
``viz/node_layer_marginals.py``, trains a classifier on per-node
features that are available immediately after the layer-0 router runs,
and reports held-out performance using a sample-level split so that
entire draft trees stay together.
"""

from __future__ import annotations

import argparse
import json
import joblib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


TOTAL_LAYER0_EXPERTS = 64.0

LAYER0_FEATURES: List[str] = [
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
        help="Optional cap on number of samples (draft trees) to use for training.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("results/combined/accept_model_metrics.json"),
        help="Where to write evaluation metrics (JSON).",
    )
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=Path("results/combined/accept_model_holdout.csv"),
        help="Optional CSV with hold-out predictions (set to '' to skip).",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("results/combined/accept_model.joblib"),
        help="Optional path to save the trained classifier (set to '' to skip).",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "accepted" not in df.columns:
        raise ValueError("Dataset missing 'accepted' column")
    return df


def build_sample_id(df: pd.DataFrame) -> pd.Series:
    keys = [
        "dataset",
        "trace_file",
        "record_index",
        "choice_index",
        "stats_index",
        "wave_index",
    ]
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns needed for sample grouping: {missing}")
    return df[keys].astype(str).agg("::".join, axis=1)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["depth"] > 0]  # root is always accepted, offers no signal

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

    def count_layer0_experts(value) -> int:
        if isinstance(value, str) and value:
            return len(value.split())
        return 0

    df["layer0_unique"] = df.get("layer0_set", "").apply(count_layer0_experts)
    df["sample_id"] = build_sample_id(df)

    # Derived structural features
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


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: HistGradientBoostingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )

    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate_test": float(y_test.mean()),
    }


def write_metrics(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(metrics, fh, indent=2)


def dump_predictions(
    df_test: pd.DataFrame,
    proba: np.ndarray,
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
        "accepted",
        "sample_id",
    ]
    available_cols = [c for c in out_cols if c in df_test.columns]
    out_df = df_test[available_cols].copy()
    out_df["accept_probability"] = proba
    out_df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    df = load_dataset(args.data)
    df = preprocess(df)
    df = subsample_samples(df, args.max_samples, args.seed)

    train_df, test_df = split_samples(df, args.test_frac, args.seed)

    X_train = train_df[LAYER0_FEATURES]
    y_train = train_df["accepted"].astype(int)

    X_test = test_df[LAYER0_FEATURES]
    y_test = test_df["accepted"].astype(int)

    model = train_classifier(X_train, y_train, args.seed)
    metrics = evaluate(model, X_test, y_test)
    write_metrics(metrics, args.metrics_out)

    if args.model_out:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_out)

    # Persist predictions if requested
    if args.predictions_out:
        proba = model.predict_proba(X_test)[:, 1]
        dump_predictions(test_df, proba, args.predictions_out)

    # Print a concise report to stdout for quick inspection
    print("Acceptance model metrics (hold-out):")
    for key, value in metrics.items():
        print(f"  {key:>18s}: {value:.4f}")

    print("\nLabel distribution (train/test):")
    print(f"  train positives: {y_train.sum()} / {len(y_train)}")
    print(f"  test positives : {y_test.sum()} / {len(y_test)}")


if __name__ == "__main__":
    main()
