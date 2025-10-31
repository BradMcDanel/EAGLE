#!/usr/bin/env python3
"""Plot wave-level gating trade-offs: expert cost vs accepted retention."""

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
        help="CSV produced by analysis/eval_wave_gating.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    baseline = df[df["lambda"].isna() & df["cost_quantile"].isna()].iloc[0]
    baseline_cost_frac = baseline["cost_kept_frac"]
    baseline_accept_frac = baseline["accepted_kept_frac"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if df["lambda"].notna().any():
        pred_df = df[(df["mode"] == "predicted") & df["lambda"].notna()].copy()
        oracle_df = df[(df["mode"] == "oracle") & df["lambda"].notna()].copy()
    else:
        pred_df = df[df["mode"] == "predicted"].copy()
        oracle_df = pd.DataFrame()

    if "lambda" in pred_df.columns:
        pred_df = pred_df.sort_values("lambda")
    else:
        pred_df = pred_df.sort_values(["cost_quantile", "pred_quantile"])
    oracle_df = oracle_df.sort_values("lambda") if not oracle_df.empty else oracle_df

    fig, ax = plt.subplots(figsize=(6, 4))
    if not pred_df.empty:
        ax.plot(
            pred_df["cost_kept_frac"] * 100,
            pred_df["accepted_kept_frac"] * 100,
            marker="o",
            linestyle="-",
            color="C0",
            linewidth=1.0,
            markersize=5,
            label="Predicted",
        )
    if not oracle_df.empty:
        ax.plot(
            oracle_df["cost_kept_frac"] * 100,
            oracle_df["accepted_kept_frac"] * 100,
            marker="o",
            linestyle="--",
            color="C1",
            linewidth=1.0,
            markersize=5,
            label="Oracle",
        )
    ax.scatter(
        baseline_cost_frac * 100,
        baseline_accept_frac * 100,
        marker="*",
        s=180,
        color="red",
        edgecolor="k",
        linewidth=0.7,
        label="Baseline",
    )
    ax.set_xlabel("Unique expert cost retained (%)")
    ax.set_ylabel("Accepted nodes retained (%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_path = args.output_dir / "wave_gating_tradeoff.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
