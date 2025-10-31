#!/usr/bin/env python3
"""Plot cumulative expert usage (count-based) curves from EAGLE expert traces."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def iter_records(path: Path) -> Iterable[Dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"
    with opener(path, mode) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iterate_token_pairs(
    expert_lists: Sequence,
    weight_lists: Sequence,
) -> Iterable[Tuple[Sequence, Sequence]]:
    """Recursively yield aligned (experts, weights) token pairs."""

    if not expert_lists or not weight_lists:
        return

    sample = expert_lists[0]
    if isinstance(sample, (list, tuple)) and sample and isinstance(sample[0], (list, tuple)):
        for sub_experts, sub_weights in zip(expert_lists, weight_lists):
            yield from _iterate_token_pairs(sub_experts, sub_weights)
    else:
        for token_experts, token_weights in zip(expert_lists, weight_lists):
            yield token_experts, token_weights


def accumulate_counts(records: Iterable[Dict]) -> Dict[int, Dict[int, int]]:
    """Accumulate selection counts per expert and per layer."""

    layer_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for record in records:
        choices = record.get("choices", [])
        if not choices:
            continue

        for turn in choices[0].get("stats", []):
            traces = turn.get("expert_traces")
            if not traces:
                continue

            for trace in traces:
                experts = trace.get("experts", {})
                for layer_str, layer_data in experts.items():
                    expert_lists = layer_data.get("experts")
                    weight_lists = layer_data.get("weights")
                    if not expert_lists or not weight_lists:
                        continue

                    for token_experts, token_weights in _iterate_token_pairs(expert_lists, weight_lists):
                        if not token_experts:
                            continue
                        for expert_id, weight in zip(token_experts, token_weights):
                            try:
                                expert_idx = int(expert_id)
                            except (TypeError, ValueError):
                                continue
                            layer_counts[int(layer_str)][expert_idx] += 1

    return layer_counts


def plot_cdfs(
    layer_counts: Dict[int, Dict[int, int]],
    output_path: Path,
    selected_layers: Optional[Iterable[int]] = None,
    title: Optional[str] = None,
) -> None:
    if not layer_counts:
        raise ValueError("No expert trace data found in the provided file.")

    if selected_layers is None:
        layers = sorted(layer_counts)
    else:
        layers = [layer for layer in sorted(layer_counts) if layer in set(selected_layers)]
        if not layers:
            raise ValueError("Selected layers not present in trace data.")

    plt.figure(figsize=(8, 5), dpi=300)

    cmap = plt.get_cmap("viridis")
    if len(layers) > 1:
        color_positions = np.linspace(0.1, 0.9, len(layers))
    else:
        color_positions = [0.5]

    for layer, color_pos in zip(layers, color_positions):
        counts = np.array(list(layer_counts[layer].values()), dtype=np.float64)
        if counts.size == 0 or counts.sum() == 0:
            continue
        counts_sorted = np.sort(counts)[::-1]
        cumulative = np.cumsum(counts_sorted) / counts_sorted.sum()
        x = np.arange(1, counts_sorted.size + 1) / counts_sorted.size
        plt.plot(
            x,
            cumulative,
            label=f"Layer {layer}",
            linewidth=1.6,
            color=cmap(color_pos),
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Uniform")

    plt.xlabel("Fraction of Experts")
    plt.ylabel("Fraction of Routed Tokens")
    plt.title(title or "Cumulative Expert Usage")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cumulative expert load (CDF) curves.")
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the MT-Bench JSONL file (optionally gzipped) containing expert traces.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices to plot (default: all layers present).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/expert_load_cdf.png"),
        help="Destination for the generated figure (default: figures/expert_load_cdf.png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer_loads = accumulate_counts(iter_records(args.input_path))

    selected_layers = None
    if args.layers:
        selected_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    plot_cdfs(layer_loads, args.output, selected_layers, args.title)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
