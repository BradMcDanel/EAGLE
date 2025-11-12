"""Simplified plots comparing flat vs shaped expert caps for OLMoE benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt

UNCAPPED_RE = re.compile(r"eagle-uncapped-t\d+-metrics\.json$")


@dataclass
class DatasetConfig:
    metric_glob: str
    baseline_glob: str
    metric_label: str
    metric_fn: Callable[[Dict], float]
    title: str
    output: Path


DATASETS: Dict[str, DatasetConfig] = {
    "gsm8k": DatasetConfig(
        metric_glob="results/gsm8k/olmoe-1b-gsm8k-eagle-*-metrics.json",
        baseline_glob="results/gsm8k/olmoe-1b-gsm8k-ar-baseline-*-metrics.json",
        metric_label="Accuracy",
        metric_fn=lambda data: float(data.get("accuracy", 0.0)),
        title="OLMoE (GSM8K) Accuracy & Speedup vs Expert Cap",
        output=Path("figures/olmoe_expert_cap_gsm8k.png"),
    ),
    "sum": DatasetConfig(
        metric_glob="results/sum/olmoe-1b-sum-eagle-*-metrics.json",
        baseline_glob="results/sum/olmoe-1b-sum-ar-baseline-*-metrics.json",
        metric_label="ROUGE-L F1",
        metric_fn=lambda data: float(data.get("rouge_l", {}).get("f1", 0.0)),
        title="OLMoE (CNN/DM) ROUGE-L & Speedup vs Expert Cap",
        output=Path("figures/olmoe_expert_cap_sum.png"),
    ),
    "alpaca": DatasetConfig(
        metric_glob="results/alpaca/olmoe-1b-alpaca-eagle-*-metrics.json",
        baseline_glob="results/alpaca/olmoe-1b-alpaca-ar-baseline-*-metrics.json",
        metric_label="F1",
        metric_fn=lambda data: float(data.get("f1", 0.0)),
        title="OLMoE (Alpaca) F1 & Speedup vs Expert Cap",
        output=Path("figures/olmoe_expert_cap_alpaca.png"),
    ),
    "humaneval": DatasetConfig(
        metric_glob="results/humaneval/olmoe-1b-humaneval-eagle-*-metrics.json",
        baseline_glob="results/humaneval/olmoe-1b-humaneval-ar-baseline-*-metrics.json",
        metric_label="pass@1",
        metric_fn=lambda data: float(data.get("pass@1", 0.0)),
        title="OLMoE (HumanEval) pass@1 & Speedup vs Expert Cap",
        output=Path("figures/olmoe_expert_cap_humaneval.png"),
    ),
}

SERIES_LABELS = {
    "shape:flat": "Flat cap shape",
    "shape:tapered": "Tapered cap shape",
    "shape:linear": "Linear cap shape",
}


def canonical_temperature(value) -> str:
    if value is None:
        return "0"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if val.is_integer():
        return str(int(val))
    return format(val, "g")


def temp_sort_key(label: str) -> float:
    try:
        return float(label)
    except (TypeError, ValueError):
        return float("inf")


def classify_series(cfg: Dict) -> Tuple[str, int]:
    schedule_file = cfg.get("cap_schedule_file")
    cap_shape = (cfg.get("cap_shape") or "").strip()
    if not cap_shape:
        # Legacy runs without per-layer shape are noisy; skip plotting them.
        return ("skip", 0)
    series_key = f"shape:{cap_shape}"
    cap_value = cfg.get("cap_shape_target")
    if cap_value is None:
        return ("skip", 0)
    return series_key, int(cap_value)


def load_runs(
    cfg: DatasetConfig,
    root: Path,
) -> Dict[str, Tuple[Dict[str, List[Tuple[int, float, float]]], Dict[str, Tuple[float, float]]]]:
    temp_map: Dict[str, Dict[str, List[Tuple[int, float, float]]]] = {}
    references: Dict[str, Dict[str, Tuple[float, float]]] = {}

    metric_paths = sorted(root.glob(cfg.metric_glob))
    if not metric_paths:
        return {}

    for path in metric_paths:
        data = json.loads(path.read_text())
        run_cfg = data.get("config", {})
        temp_key = canonical_temperature(run_cfg.get("temperature"))
        metric_val = cfg.metric_fn(data)
        throughput = float(data.get("generation_stats", {}).get("mean_throughput", 0.0))

        if UNCAPPED_RE.search(path.name):
            references.setdefault(temp_key, {})["EAGLE3"] = (metric_val, throughput)
            continue

        series_key, cap_value = classify_series(run_cfg)
        if series_key == "skip":
            continue

        temp_map.setdefault(temp_key, {}).setdefault(series_key, []).append((cap_value, metric_val, throughput))

    for baseline_path in root.glob(cfg.baseline_glob):
        data = json.loads(baseline_path.read_text())
        temp_key = canonical_temperature(data.get("config", {}).get("temperature"))
        references.setdefault(temp_key, {})["AR baseline"] = (
            cfg.metric_fn(data),
            float(data.get("generation_stats", {}).get("mean_throughput", 0.0)),
        )

    grouped: Dict[str, Tuple[Dict[str, List[Tuple[int, float, float]]], Dict[str, Tuple[float, float]]]] = {}
    for temp_key, series in temp_map.items():
        cleaned = {key: sorted(points, key=lambda x: x[0]) for key, points in series.items() if points}
        if cleaned:
            grouped[temp_key] = (cleaned, references.get(temp_key, {}))
    return grouped


def plot_dataset(
    config: DatasetConfig,
    temp_key: str,
    series_map: Dict[str, List[Tuple[int, float, float]]],
    references: Dict[str, Tuple[float, float]],
    output_path: Path,
) -> None:
    baseline = references.get("AR baseline")
    if baseline is None or baseline[0] == 0 or baseline[1] == 0:
        raise ValueError(f"Missing AR baseline for temp={temp_key}")

    caps_present = sorted({cap for points in series_map.values() for cap, _, _ in points})
    if not caps_present:
        raise ValueError(f"No sweep points for {config.title} @ temp={temp_key}")

    fig, (ax_metric, ax_speed) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue", "tab:orange", "tab:green"])

    legend_handles = []
    legend_labels = []
    for idx, (series_key, points) in enumerate(sorted(series_map.items(), key=lambda item: item[0])):
        label = SERIES_LABELS.get(series_key, series_key)
        color = colors[idx % len(colors)]
        caps = [cap for cap, _, _ in points]
        rel_metric = [metric / baseline[0] for _, metric, _ in points]
        speedup = [thr / baseline[1] for _, _, thr in points]
        ax_metric.plot(caps, rel_metric, marker="o", color=color, linewidth=2, label=label)
        handle = ax_speed.plot(caps, speedup, marker="o", color=color, linewidth=2, label=label)[0]
        legend_handles.append(handle)
        legend_labels.append(label)

    uncapped = references.get("EAGLE3")
    if uncapped:
        rel_metric = uncapped[0] / baseline[0]
        speedup = uncapped[1] / baseline[1]
        ax_metric.scatter([max(caps_present) + 2], [rel_metric], marker="D", color="tab:purple", label="EAGLE3")
        ax_speed.scatter([max(caps_present) + 2], [speedup], marker="D", color="tab:purple", label="EAGLE3")

    ax_metric.set_ylabel(f"Relative {config.metric_label} vs AR")
    ax_metric.grid(True, alpha=0.3)
    ax_metric.tick_params(labelbottom=False)

    ax_speed.set_xlabel("Average Expert Cap")
    ax_speed.set_ylabel("Speedup vs AR Baseline")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(legend_handles, legend_labels, loc="lower left", framealpha=0.9)

    fig.suptitle(f"{config.title} (temp={temp_key})", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OLMoE expert-cap sweeps.")
    parser.add_argument("--results-root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS.keys()),
        help="Comma-separated dataset names (default: all).",
    )
    args = parser.parse_args()

    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not selected:
        selected = list(DATASETS.keys())
    for name in selected:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset '{name}'")

    for name in selected:
        cfg = DATASETS[name]
        grouped = load_runs(cfg, args.results_root)
        if not grouped:
            print(f"Skipping {name}: no sweep data found.")
            continue
        for temp_key in sorted(grouped.keys(), key=temp_sort_key):
            series_map, references = grouped[temp_key]
            output_path = cfg.output.with_name(f"{cfg.output.stem}_temp{temp_key}{cfg.output.suffix}")
            plot_dataset(cfg, temp_key, series_map, references, output_path)
            print(f"Wrote plot to {output_path} (temp={temp_key})")


if __name__ == "__main__":
    main()
