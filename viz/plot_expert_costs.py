#!/usr/bin/env python3
"""Generate visualizations comparing expert cost distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def add_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    denom = df["accepted_len"].clip(lower=1)
    df = df.copy()
    df["experts_per_token"] = df["accepted_unique_total"] / denom

    weight_cols = [col for col in df.columns if col.endswith("_accepted_weight")]
    if weight_cols:
        df["accepted_weight_total"] = df[weight_cols].sum(axis=1)
        df["accepted_weight_per_token"] = df["accepted_weight_total"] / denom
    else:
        df["accepted_weight_per_token"] = np.nan
    return df


def boxplot_by_depth(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    depths_sorted = sorted(df["depth"].unique())
    legend_handles = []
    legend_labels = []

    for idx, (dataset, group) in enumerate(df.groupby("dataset")):
        data = []
        positions = []
        offset = -0.2 if idx == 0 else 0.2
        for pos, depth in enumerate(depths_sorted):
            vals = group[group["depth"] == depth]["experts_per_token"].dropna()
            if len(vals) == 0:
                continue
            data.append(vals)
            positions.append(pos + offset)

        if not data:
            continue

        color = f"C{idx}"
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.35,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.4),
            medianprops=dict(color="black"),
            showfliers=False,
        )
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=6))
        legend_labels.append(dataset)

    ax.set_xticks(range(len(depths_sorted)))
    ax.set_xticklabels(depths_sorted)
    ax.set_xlabel("Tree Depth")
    ax.set_ylabel("Unique Experts per Accepted Token")
    ax.set_title("Expert Cost by Depth")
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper right")
    fig.tight_layout()
    ensure_output_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)


def cdf_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, (dataset, group) in enumerate(df.groupby("dataset")):
        values = np.sort(group["experts_per_token"].dropna())
        if len(values) == 0:
            continue
        cdf = (np.arange(len(values)) + 1) / len(values)
        ax.plot(values, cdf, label=dataset, color=f"C{idx}")

    ax.set_xlabel("Unique Experts per Accepted Token")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of Expert Cost")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    ensure_output_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)


def scatter_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for idx, (dataset, group) in enumerate(df.groupby("dataset")):
        ax.scatter(
            group["accepted_len"],
            group["accepted_unique_total"],
            alpha=0.4,
            s=10,
            label=dataset,
            color=f"C{idx}",
        )

    ax.set_xlabel("Accepted Tokens")
    ax.set_ylabel("Total Unique Experts")
    ax.set_title("Accepted Utility vs. Expert Cost (per draft/verify cycle)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    ensure_output_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)


def lorenz_efficiency(df: pd.DataFrame, out_path: Path, num_bootstrap: int = 1000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    curves = []

    base = df.copy()
    base["accepted_len"] = base["accepted_len"].clip(lower=0)
    base["accepted_unique_total"] = base["accepted_unique_total"].clip(lower=0)
    base = base[base["accepted_len"] > 0]
    base = base.sort_values("experts_per_token")

    def cumulative_curve(values: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        experts = values["accepted_unique_total"].to_numpy(dtype=float)
        tokens = values["accepted_len"].to_numpy(dtype=float)
        cum_experts = experts.cumsum()
        cum_tokens = tokens.cumsum()
        cum_experts /= cum_experts[-1]
        cum_tokens /= cum_tokens[-1]
        return cum_experts, cum_tokens

    base_curve = cumulative_curve(base)

    for _ in range(num_bootstrap):
        sample = base.sample(frac=1.0, replace=True, random_state=rng.integers(1 << 32))
        curves.append(cumulative_curve(sample))

    xs = base_curve[0]
    ys_boot = np.vstack([curve[1] for curve in curves])
    lower = np.percentile(ys_boot, 2.5, axis=0)
    upper = np.percentile(ys_boot, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(xs, base_curve[1], label="Observed", color="C0")
    ax.fill_between(xs, lower, upper, color="C0", alpha=0.2, label="95% CI")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect efficiency")
    ax.set_xlabel("Cumulative share of expert cost")
    ax.set_ylabel("Cumulative share of accepted tokens")
    ax.set_title("Cumulative Efficiency (Lorenz curve)")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    ensure_output_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)


def accepted_experts_per_cycle(
    df: pd.DataFrame,
    out_path: Path,
    max_cycle: int | None = 60,
) -> None:
    cycles = df.copy()
    cycles = cycles.dropna(subset=["accepted_unique_total", "layer_count"])
    cycles = cycles[cycles["layer_count"] > 0]
    if cycles.empty:
        return

    cycles["accepted_per_layer"] = cycles["accepted_unique_total"] / cycles["layer_count"]

    grouped = cycles.groupby(["dataset", "iteration"])
    stats = grouped["accepted_per_layer"].agg(
        mean="mean",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        count="count",
    ).reset_index()

    if max_cycle is not None:
        stats = stats[stats["iteration"] <= max_cycle]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for idx, (dataset, group) in enumerate(stats.groupby("dataset")):
        group = group.sort_values("iteration")
        color = f"C{idx}"
        ax.plot(group["iteration"], group["mean"], label=f"{dataset}", color=color)
        ax.fill_between(
            group["iteration"],
            group["q25"],
            group["q75"],
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel("Draft/Verify Cycle Index")
    ax.set_ylabel("Accepted experts per layer")
    ax.set_title("Accepted-path expert usage over draft/verify cycles")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    ensure_output_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot expert cost distributions")
    parser.add_argument("waves", type=Path, help="CSV file with per-cycle metrics")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/combined/plots"), help="Directory for plot outputs"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.waves)
    df = add_cost_columns(df)

    boxplot_by_depth(df, args.out_dir / "experts_per_token_by_depth.png")
    cdf_plot(df, args.out_dir / "experts_per_token_cdf.png")
    scatter_plot(df, args.out_dir / "accepted_vs_experts.png")
    lorenz_efficiency(df, args.out_dir / "cumulative_efficiency.png")
    accepted_experts_per_cycle(df, args.out_dir / "accepted_experts_per_cycle.png")


if __name__ == "__main__":
    main()
