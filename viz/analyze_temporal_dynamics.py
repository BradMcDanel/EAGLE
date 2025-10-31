#!/usr/bin/env python3
"""Analyze temporal trends of draft-tree cost and utility."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_series(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset, group in df.groupby("dataset"):
        series = group.sort_values("iteration").reset_index(drop=True)
        series_path = out_dir / f"temporal_series_{dataset}.csv"
        series.to_csv(series_path, index=False)

        corr_accept = series["iteration"].corr(series["accepted_len"])
        corr_layer0 = series["iteration"].corr(series["layer0_unique"])

        print(f"Dataset {dataset}:")
        print(f"  iterations: {series['iteration'].min()} -> {series['iteration'].max()}")
        print(f"  mean accepted length: {series['accepted_len'].mean():.3f}")
        print(f"  mean layer0 unique experts: {series['layer0_unique'].mean():.3f}")
        print(f"  corr(iteration, accepted_len) = {corr_accept:.3f}")
        print(f"  corr(iteration, layer0_unique) = {corr_layer0:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal analysis for draft tree cost")
    parser.add_argument("cycle_csv", type=Path, help="CSV produced by analyze_tree_structure.py (cycle_metrics.csv)")
    parser.add_argument("--out-dir", type=Path, default=Path("results/combined/temporal"))
    args = parser.parse_args()

    df = pd.read_csv(args.cycle_csv)
    required = {"dataset", "iteration", "accepted_len", "layer0_unique"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in cycle metrics: {missing}")

    compute_series(df, args.out_dir)


if __name__ == "__main__":
    main()

