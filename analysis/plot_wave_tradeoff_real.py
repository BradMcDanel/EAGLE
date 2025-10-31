#!/usr/bin/env python3
"""Plot mean experts per tree vs acceptance for different lambda settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with real metrics (from compute_wave_gating_metrics.py)")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--title", type=str, default="Wave Gating Trade-off (Real Units)")
    parser.add_argument("--xlabel", type=str, default="Mean unique experts per tree")
    parser.add_argument("--ylabel", type=str, default="Mean wave acceptance rate")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    baseline = df[df["mode"] == "baseline"].iloc[0]
    predicted_df = df[df["mode"] == "predicted"].sort_values("lambda")
    oracle_df = df[df["mode"] == "oracle"].sort_values("lambda")

    fig, ax = plt.subplots(figsize=(6, 4))
    x_col_pred = "mean_unique_experts_per_tree"
    y_col_pred = "mean_accepted_per_tree"
    x_col_base = "baseline_mean_unique_experts_per_tree"
    y_col_base = "baseline_mean_accepted_per_tree"

    if not predicted_df.empty:
        ax.plot(
            predicted_df[x_col_pred],
            predicted_df[y_col_pred],
            marker="o",
            linestyle="-",
            color="C0",
            linewidth=1.0,
            markersize=5,
            label="Predicted",
        )
    if not oracle_df.empty:
        ax.plot(
            oracle_df[x_col_pred],
            oracle_df[y_col_pred],
            marker="o",
            linestyle="--",
            color="C1",
            linewidth=1.0,
            markersize=5,
            label="Oracle",
        )

    ax.scatter(
        baseline[x_col_base],
        baseline[y_col_base],
        marker="*",
        s=180,
        color="red",
        edgecolor="k",
        linewidth=0.7,
        label="Baseline",
    )

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    ax.legend(loc="best")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved real-units trade-off to {args.output}")


if __name__ == "__main__":
    main()
