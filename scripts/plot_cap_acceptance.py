#!/usr/bin/env python3
"""
Visualize how fixed expert caps affect acceptance length and tokens-per-iteration.

Reads the summary CSV produced by `scripts/run_eagle_cap_sweep.sh`, groups rows by
dataset and temperature, and plots mean acceptance length (x-axis) versus mean tokens
per iteration (y-axis). Each point is annotated with the cap value so we can verify
that lower caps indeed reduce acceptance while higher caps restore it.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_cap(raw_value: str) -> float | None:
    """Extract the numeric cap from a string like 'cap12', 'flat-28', or '36'."""
    if not raw_value:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", raw_value)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if value.is_integer():
        return int(value)
    return value


def load_summary(csv_path: Path) -> Dict[Tuple[str, str], List[Tuple[int, float, float]]]:
    """
    Return a mapping from (dataset, temperature) to a list of
    (cap, mean_accept_length, mean_tokens_per_iter).
    """
    grouped: DefaultDict[Tuple[str, str], List[Tuple[int, float, float]]] = DefaultDict(list)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cap_str = row.get("expert_cap") or ""
            cap_mode = (row.get("cap_mode") or "static").strip().lower()
            cap = parse_cap(cap_str)
            if cap is None and cap_mode == "fidelity":
                try:
                    cap = float(row.get("mean_active_experts"))
                except (TypeError, ValueError):
                    cap = None
            if cap is None:
                continue
            # Ignore rows where the controller was in adaptive or schedule modes.
            mode = (row.get("mode") or "").lower()
            if "adaptive" in mode or "schedule" in mode:
                continue
            try:
                mean_accept = float(row["mean_accept_length"])
                mean_tokens = float(row["mean_tokens_per_iter"])
            except (KeyError, ValueError):
                continue
            dataset = row.get("dataset") or "unknown"
            temperature = row.get("temperature") or "0"
            grouped[(dataset, temperature)].append((cap, mean_accept, mean_tokens))
    # Sort each list by cap ascending for cleaner plotting.
    for key, values in grouped.items():
        values.sort(key=lambda item: item[0])
    return grouped


def make_plots(
    grouped: Dict[Tuple[str, str], List[Tuple[int, float, float]]],
    output_path: Path,
) -> None:
    datasets = sorted({key[0] for key in grouped})
    if not datasets:
        raise SystemExit("No static cap data found in the provided CSV.")

    ncols = 2
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    for ax, dataset in zip(axes.flat, datasets):
        temps = sorted({key[1] for key in grouped if key[0] == dataset})
        for temp in temps:
            points = grouped[(dataset, temp)]
            if not points:
                continue
            accepts = [p[1] for p in points]
            tokens = [p[2] for p in points]
            caps = [p[0] for p in points]
            ax.plot(
                accepts,
                tokens,
                marker="o",
                linestyle="-",
                label=f"T={temp}",
            )
            for accept, tok, cap in zip(accepts, tokens, caps):
                ax.annotate(
                    str(cap),
                    xy=(accept, tok),
                    xytext=(4, 2),
                    textcoords="offset points",
                    fontsize=8,
                    color="gray",
                )
        ax.set_title(dataset)
        ax.set_xlabel("Mean accept length")
        ax.set_ylabel("Mean tokens per iter")
        if temps:
            ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide any unused subplots.
    total_axes = nrows * ncols
    for idx in range(len(datasets), total_axes):
        axes.flat[idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Wrote plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot acceptance vs. tokens for flat cap runs.")
    parser.add_argument(
        "summary_csv",
        type=Path,
        help="Path to results/summaries/cap_sweep.csv (or similar).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cap_accept_vs_tokens.png"),
        help="Where to write the PNG plot (default: %(default)s).",
    )
    args = parser.parse_args()

    grouped = load_summary(args.summary_csv)
    make_plots(grouped, args.output)


if __name__ == "__main__":
    main()
