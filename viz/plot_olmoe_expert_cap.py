"""Plot accuracy/throughput vs expert cap for OLMoE across benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OLD_CAP_PATTERN = re.compile(r"olmoe-1b-eagle_B(\d+)-question-metrics\.json")
NEW_CAP_PATTERN = re.compile(r"olmoe-1b-[^-]+-eagle-cap(\d+)-metrics\.json")
UNCAPPED_PATTERN = re.compile(r"olmoe-1b-[^-]+-eagle-uncapped-metrics\.json")
AR_BASELINE_PATTERN = re.compile(r"olmoe-1b-[^-]+-ar-baseline-metrics\.json")

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
        metric_glob="results/gsm8k/olmoe-1b*-metrics.json",
        baseline_file=Path("results/gsm8k/olmoe-1b-gsm8k-ar-baseline-metrics.json"),
        metric_label="Accuracy",
        metric_fn=lambda data: float(data.get("accuracy", 0.0)),
        title="OLMoE (GSM8K) Accuracy & Speedup vs Expert Cap",
        output=Path("figures/olmoe_expert_cap_gsm8k.png"),
    ),
    "sum": DatasetConfig(
        name="sum",
        metric_glob="results/sum/olmoe-1b*-metrics.json",
        baseline_file=Path("results/sum/olmoe-1b-sum-ar-baseline-metrics.json"),
        metric_label="ROUGE-L F1",
        metric_fn=lambda data: float(data.get("rouge_l", {}).get("f1", 0.0)),
        title="OLMoE (CNN/DM) ROUGE-L & Speedup vs Expert Cap",
        output=Path("figures/olmoe_expert_cap_sum.png"),
    ),
    "alpaca": DatasetConfig(
        name="alpaca",
        metric_glob="results/alpaca/olmoe-1b*-metrics.json",
        baseline_file=Path("results/alpaca/olmoe-1b-alpaca-ar-baseline-metrics.json"),
        metric_label="F1",
        metric_fn=lambda data: float(data.get("f1", 0.0)),
        title="OLMoE (Alpaca) F1 & Speedup vs Expert Cap",
        output=Path("figures/olmoe_expert_cap_alpaca.png"),
    ),
}


def load_metrics(config: DatasetConfig, root: Path) -> Tuple[List[Tuple[int, float, float]], Dict[str, Tuple[float, float]]]:
    """Return sweep points and reference markers for a dataset."""

    sweep_points: List[Tuple[int, float, float]] = []
    references: Dict[str, Tuple[float, float]] = {}

    for path in root.glob(config.metric_glob):
        name = path.name
        match_old = OLD_CAP_PATTERN.match(name)
        match_new = NEW_CAP_PATTERN.match(name)

        data: Dict = json.loads(path.read_text())
        metric_val = config.metric_fn(data)
        throughput = float(data.get("generation_stats", {}).get("mean_throughput", 0.0))

        if match_old:
            sweep_points.append((int(match_old.group(1)), metric_val, throughput))
        elif match_new:
            sweep_points.append((int(match_new.group(1)), metric_val, throughput))
        elif UNCAPPED_PATTERN.match(name):
            references["EAGLE3"] = (metric_val, throughput)
        elif AR_BASELINE_PATTERN.match(name):
            references["AR baseline"] = (metric_val, throughput)

    baseline_path = (root / config.baseline_file).resolve()
    if baseline_path.exists() and "AR baseline" not in references:
        data: Dict = json.loads(baseline_path.read_text())
        references["AR baseline"] = (
            config.metric_fn(data),
            float(data.get("generation_stats", {}).get("mean_throughput", 0.0)),
        )

    sweep_points.sort(key=lambda x: x[0])
    return sweep_points, references


def plot(data: Tuple[List[Tuple[int, float, float]], Dict[str, Tuple[float, float]]], config: DatasetConfig, output_path: Path) -> None:
    sweep_points, references = data
    if not sweep_points:
        raise ValueError(f"No OLMoE metrics found for {config.name}. Did you run the benchmark?")

    capped_points = [(c, a, t) for c, a, t in sweep_points if isinstance(c, int) and c <= 36]
    if not capped_points:
        raise ValueError(f"No capped expert runs (<=36) found for {config.name}.")

    capped_points.sort(key=lambda x: x[0])
    caps = [p[0] for p in capped_points]
    metrics = [p[1] for p in capped_points]
    throughput = [p[2] for p in capped_points]

    # Get AR baseline values for relative calculations
    ar_baseline_metric, ar_baseline_throughput = references.get("AR baseline", (None, None))
    if ar_baseline_throughput is None or ar_baseline_throughput == 0:
        raise ValueError("AR baseline throughput not found or is zero. Cannot calculate speedup.")
    if ar_baseline_metric is None or ar_baseline_metric == 0:
        raise ValueError("AR baseline metric not found or is zero. Cannot calculate relative metric.")

    # Calculate relative metrics
    relative_metric = [m / ar_baseline_metric for m in metrics]
    speedup = [t / ar_baseline_throughput for t in throughput]

    has_uncapped = "EAGLE3" in references
    eagle_metric = eagle_speed = None
    if has_uncapped:
        metric_val, thr_val = references["EAGLE3"]
        eagle_metric = metric_val / ar_baseline_metric if metric_val is not None else None
        eagle_speed = thr_val / ar_baseline_throughput if thr_val is not None else None
        if eagle_metric is None or eagle_speed is None:
            has_uncapped = False

    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 1], hspace=0.05, wspace=0.05)

    ax1_left = fig.add_subplot(gs[0, 0])
    ax2_left = fig.add_subplot(gs[1, 0], sharex=ax1_left)

    ax1_right = fig.add_subplot(gs[0, 1], sharey=ax1_left) if has_uncapped else None
    ax2_right = fig.add_subplot(gs[1, 1], sharey=ax2_left) if has_uncapped else None

    sweep_color = "tab:blue"

    # Relative metric plot (left)
    line_metric, = ax1_left.plot(caps, relative_metric, marker="o", color=sweep_color, linewidth=2, label="Expert Cap")
    ax1_left.set_ylabel(f"Relative {config.metric_label} vs AR")
    ax1_left.grid(True, alpha=0.3)
    x_padding = (max(caps) - min(caps)) * 0.02
    ax1_left.set_xlim(min(caps) - x_padding, max(caps) + x_padding)
    ax1_left.set_xticks(caps)
    ax1_left.tick_params(labelbottom=False)
    ax1_left.spines['right'].set_visible(False)
    ax1_left.spines['top'].set_visible(False)
    ax1_left.tick_params(right=False, labelright=False)

    metric_vals = relative_metric + [1.0]
    if has_uncapped:
        ax1_right.plot([64], [eagle_metric], marker="D", color="tab:purple", markersize=7,
                       markeredgewidth=1.2, markeredgecolor="black", label="EAGLE3")
        ax1_right.set_xlim(63.5, 64.5)
        ax1_right.set_xticks([64])
        ax1_right.set_xticklabels(["64\n(Uncapped)"])
        ax1_right.grid(True, alpha=0.3)
        ax1_right.tick_params(labelleft=False, left=False, labelbottom=False)
        ax1_right.spines['left'].set_visible(False)
        ax1_right.spines['right'].set_visible(False)
        ax1_right.spines['top'].set_visible(False)
        metric_vals.append(eagle_metric)

    ax1_left.margins(y=0.1)
    if has_uncapped:
        ax1_right.margins(y=0.1)

    # Speedup plot (left)
    line_speed, = ax2_left.plot(caps, speedup, marker="o", color=sweep_color, linewidth=2, label="Expert Cap")
    ax2_left.set_xlabel("Expert Cap")
    ax2_left.set_ylabel("Speedup vs AR Baseline")
    ax2_left.grid(True, alpha=0.3)
    ax2_left.set_xlim(min(caps) - x_padding, max(caps) + x_padding)
    ax2_left.set_xticks(caps)
    ax2_left.spines['right'].set_visible(False)
    ax2_left.spines['top'].set_visible(False)
    ax2_left.tick_params(right=False, labelright=False)

    speed_vals = speedup[:]
    if has_uncapped:
        ax2_right.plot([64], [eagle_speed], marker="D", color="tab:purple", markersize=7,
                       markeredgewidth=1.2, markeredgecolor="black", label="EAGLE3")
        ax2_right.set_xlim(63.5, 64.5)
        ax2_right.set_xticks([64])
        ax2_right.set_xticklabels(["64\n(Uncapped)"])
        ax2_right.grid(True, alpha=0.3)
        ax2_right.tick_params(labelleft=False, left=False)
        ax2_right.spines['left'].set_visible(False)
        ax2_right.spines['right'].set_visible(False)
        ax2_right.spines['top'].set_visible(False)
        speed_vals.append(eagle_speed)

    ax2_left.margins(y=0.1)
    if has_uncapped:
        ax2_right.set_ylim(ax2_left.get_ylim())

    if has_uncapped:
        ax2_right.set_ylim(ax2_left.get_ylim())

    # Only show legend on bottom subplot
    handles = [line_speed]
    labels = ["Expert Cap"]
    if has_uncapped:
        eagle_handle = Line2D([0], [0], marker="D", color="tab:purple", markersize=7,
                              markeredgewidth=1.2, markeredgecolor="black", linestyle="none")
        handles.append(eagle_handle)
        labels.append("EAGLE3")
    ax2_left.legend(handles, labels, loc="lower left", framealpha=0.9)

    fig.suptitle(config.title, fontsize=13)

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
        try:
            points = load_metrics(cfg, args.results_root)
            output = cfg.output
            output.parent.mkdir(parents=True, exist_ok=True)
            plot(points, cfg, output)
            print(f"Wrote plot to {output}")
        except ValueError as exc:
            print(f"Skipping {cfg.name}: {exc}")


if __name__ == "__main__":
    main()
