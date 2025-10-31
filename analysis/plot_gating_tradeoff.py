#!/usr/bin/env python3
"""Plot expert-count vs accepted-loss trade-offs from a sweep CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV produced by analysis/eval_gating_tradeoff.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where figures will be saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    baseline = df[df["accept_threshold"].isna()].iloc[0]
    baseline_expert_ratio = baseline["kept_expert_ratio"]
    baseline_accept_ratio = baseline["kept_accept_ratio"]

    plot_df = df.dropna(subset=["accept_threshold"]).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(
        plot_df["kept_expert_ratio"] * 100,
        plot_df["kept_accept_ratio"] * 100,
        c=plot_df["cost_quantile"],
        cmap="viridis",
        s=70,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.scatter(
        baseline_expert_ratio * 100,
        baseline_accept_ratio * 100,
        marker="*",
        s=180,
        color="red",
        edgecolor="k",
        linewidth=0.7,
        label="Baseline",
    )
    ax.set_xlabel("Unique experts retained (%)")
    ax.set_ylabel("Acceptance rate retained (%)")
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Cost quantile")

    fig.tight_layout()
    output_path = args.output_dir / "gating_expert_vs_accept.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
