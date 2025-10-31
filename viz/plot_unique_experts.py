#!/usr/bin/env python3
"""Visualize mean unique experts per layer from EAGLE expert traces."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple, Union

import matplotlib.pyplot as plt


def iter_records(path: Path) -> Iterable[Dict]:
    """Yield JSON records from a .jsonl or .jsonl.gz file."""

    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"

    with opener(path, mode) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_expert_ids(value: Union[int, Iterable]) -> Iterator[int]:
    """Yield integers from potentially nested expert id collections."""

    if isinstance(value, int):
        yield value
    elif isinstance(value, (str, bytes)):
        yield int(value)
    elif isinstance(value, Iterable):
        for item in value:
            yield from _iter_expert_ids(item)


def accumulate_unique_experts(records: Iterable[Dict]) -> Dict[int, Tuple[int, int]]:
    """Accumulate total unique expert counts per layer across iterations."""

    totals: Dict[int, Tuple[int, int]] = defaultdict(lambda: [0, 0])

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
                    if not expert_lists:
                        continue

                    unique_experts = set()
                    for token_experts in expert_lists:
                        for expert_id in _iter_expert_ids(token_experts):
                            unique_experts.add(int(expert_id))

                    layer_idx = int(layer_str)
                    totals[layer_idx][0] += len(unique_experts)
                    totals[layer_idx][1] += 1

    return totals


def plot_unique_experts(layer_totals: Dict[int, Tuple[int, int]], output_path: Path) -> None:
    """Generate a bar chart of mean unique experts per layer."""

    if not layer_totals:
        raise ValueError("No expert trace data found in the provided file.")

    layers = sorted(layer_totals)
    means = [layer_totals[layer][0] / layer_totals[layer][1] for layer in layers]

    plt.figure(figsize=(10, 5), dpi=300)
    bars = plt.bar(layers, means, color="#4263eb")

    plt.xlabel("Layer Index")
    plt.ylabel("Mean Unique Experts")
    plt.title("Mean Unique Experts per Layer")
    plt.xticks(layers)

    for bar, mean_val in zip(bars, means):
        bar.set_edgecolor("black")
        bar.set_linewidth(0.5)
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mean unique experts per layer from expert traces.")
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the MT-Bench JSONL file (optionally gzipped) containing expert traces.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/mean_unique_experts.png"),
        help="Destination for the generated figure (default: figures/mean_unique_experts.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer_totals = accumulate_unique_experts(iter_records(args.input_path))
    plot_unique_experts(layer_totals, args.output)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
