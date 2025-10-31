#!/usr/bin/env python3
"""Plot marginal expert additions per depth from node-level data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot marginal experts per depth")
    parser.add_argument("node_csv", type=Path, help="CSV from node_layer_marginals.py")
    parser.add_argument("--column", type=str, default="layer_0_marginal", help="Column containing marginal counts (e.g., layer_0_marginal or total_marginal)")
    parser.add_argument("--out", type=Path, default=Path("results/combined/plots/marginal_experts_by_depth.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.node_csv)
    if df.empty:
        print("Empty node dataset")
        return

    if args.column not in df.columns:
        raise ValueError(f"Column {args.column} not found in node data")

    grouped = df.groupby(["dataset", "depth"])
    stats = grouped[args.column].agg([
        ("mean", "mean"),
        ("q25", lambda x: x.quantile(0.25)),
        ("q75", lambda x: x.quantile(0.75)),
    ]).reset_index()

    datasets = stats["dataset"].unique()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for idx, dataset in enumerate(datasets):
        subset = stats[stats["dataset"] == dataset]
        ax.plot(subset["depth"], subset["mean"], label=f"{dataset}", color=f"C{idx}")
        ax.fill_between(
            subset["depth"],
            subset["q25"],
            subset["q75"],
            color=f"C{idx}",
            alpha=0.2,
        )

    ax.set_xlabel("Depth")
    ax.set_ylabel(f"Marginal experts per node ({args.column})")
    ax.set_title(f"Average marginal expert additions vs depth ({args.column})")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
