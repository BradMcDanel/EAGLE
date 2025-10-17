"""Plot accuracy/throughput vs expert cap for OLMoE across benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


PATTERN = re.compile(r"olmoe-1b-eagle_B(\d+)-question-metrics\.json")
BUDGET_CAP = 64  # Max expert budget for plotting / baseline marker location


@dataclass
class DatasetConfig:
    name: str
    metric_glob: str
    baseline_file: Path
    metric_label: str
    metric_fn: Callable[[Dict], float]
    title: str
    output: Path


DATASETS: Dict[str, DatasetConfig] = {
    "gsm8k": DatasetConfig(
        name="gsm8k",
        metric_glob="results/gsm8k/olmoe-1b-eagle_B*-question-metrics.json",
        baseline_file=Path("results/gsm8k/olmoe-1b-baseline-question-metrics.json"),
        metric_label="Accuracy",
        metric_fn=lambda data: float(data.get("accuracy", 0.0)),
        title="OLMoE (GSM8K) Accuracy & Throughput vs Expert Cap",
        output=Path("figures/olmoe_expert_cap_gsm8k.png"),
    ),
    "sum": DatasetConfig(
        name="sum",
        metric_glob="results/sum/olmoe-1b-eagle_B*-question-metrics.json",
        baseline_file=Path("results/sum/olmoe-1b-baseline-question-metrics.json"),
        metric_label="ROUGE-L F1",
        metric_fn=lambda data: float(data.get("rouge_l", {}).get("f1", 0.0)),
        title="OLMoE (SUM) ROUGE-L & Throughput vs Expert Cap",
        output=Path("figures/olmoe_expert_cap_sum.png"),
    ),
}


def load_metrics(config: DatasetConfig, root: Path) -> List[Tuple[int, float, float, str]]:
    """Return (expert_cap, metric, throughput, label) sorted by cap."""

    points: List[Tuple[int, float, float, str]] = []
    for path in root.glob(config.metric_glob):
        match = PATTERN.match(path.name)
        if not match:
            continue

        cap = int(match.group(1))
        data: Dict = json.loads(path.read_text())
        accuracy = config.metric_fn(data)
        throughput = float(
            data.get("generation_stats", {}).get("mean_throughput", 0.0)
        )
        points.append((cap, accuracy, throughput, f"EAGLE B{cap}"))

    # Append baseline if available
    baseline_path = (root / config.baseline_file).resolve()
    if baseline_path.exists():
        data: Dict = json.loads(baseline_path.read_text())
        points.append(
            (
                BUDGET_CAP,
                config.metric_fn(data),
                float(data.get("generation_stats", {}).get("mean_throughput", 0.0)),
                "Baseline",
            )
        )

    # Move expert cap 0 to the end (treated as full-budget run)
    ordered = sorted(points, key=lambda x: (x[0] == 0, x[0]))

    transformed: List[Tuple[int, float, float, str]] = []
    for cap, acc, thr, label in ordered:
        cap_plot = BUDGET_CAP if cap == 0 else cap
        transformed.append((cap_plot, acc, thr, label))

    # ensure unique by expert cap ordering (baseline already assigned BUDGET_CAP)
    transformed.sort(key=lambda x: x[0])
    return transformed


def plot(points: List[Tuple[int, float, float, str]], config: DatasetConfig, output_path: Path) -> None:
    if not points:
        raise ValueError(f"No OLMoE metrics found for {config.name}. Did you run the benchmark?")

    caps = [p[0] for p in points]
    accuracy = [p[1] for p in points]
    throughput = [p[2] for p in points]
    labels = [p[3] for p in points]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Expert Cap")
    ax1.set_ylabel(config.metric_label, color="tab:blue")

    line_caps = [c for c, lbl in zip(caps, labels) if lbl != "Baseline"]
    line_acc = [a for a, lbl in zip(accuracy, labels) if lbl != "Baseline"]
    if line_caps:
        ax1.plot(line_caps, line_acc, marker="o", color="tab:blue", label="Accuracy")

    baseline_caps = [c for c, lbl in zip(caps, labels) if lbl == "Baseline"]
    baseline_acc = [a for a, lbl in zip(accuracy, labels) if lbl == "Baseline"]
    if baseline_caps:
        ax1.scatter(baseline_caps, baseline_acc, marker="*", s=140, color="tab:blue", label="Baseline")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Throughput (tokens/s)", color="tab:red")

    if line_caps:
        ax2.plot(line_caps, [t for t, lbl in zip(throughput, labels) if lbl != "Baseline"], marker="s", color="tab:red", label="Throughput")
    if baseline_caps:
        baseline_thr = [t for t, lbl in zip(throughput, labels) if lbl == "Baseline"]
        ax2.scatter(baseline_caps, baseline_thr, marker="*", s=140, color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle(config.title, fontsize=12)

    lines_labels = [ax.get_legend_handles_labels() for ax in (ax1, ax2)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc="best", ncol=2, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot metric/throughput vs expert cap for OLMoE benchmarks"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("."),
        help="Repository root (defaults to current directory)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS.keys()),
        help=f"Comma-separated benchmark names to plot (default: {','.join(DATASETS.keys())})",
    )
    args = parser.parse_args()

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in requested if d not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")

    for key in requested:
        cfg = DATASETS[key]
        points = load_metrics(cfg, args.results_root)
        output = cfg.output
        output.parent.mkdir(parents=True, exist_ok=True)
        plot(points, cfg, output)
        print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
