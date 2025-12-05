"""Plot pruning strategy comparison (substitution vs truncation) for OLMoE benchmarks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt


@dataclass
class DatasetConfig:
    metric_label: str
    metric_fn: Callable[[Dict], float]
    title: str
    output: Path


DATASETS: Dict[str, DatasetConfig] = {
    "gsm8k": DatasetConfig(
        metric_label="Accuracy",
        metric_fn=lambda data: float(data.get("accuracy", 0.0)),
        title="OLMoE (GSM8K) Accuracy & Speedup vs Expert Cap",
        output=Path("figures/pruning_strategies_gsm8k.png"),
    ),
    "sum": DatasetConfig(
        metric_label="ROUGE-L F1",
        metric_fn=lambda data: float(data.get("rouge_l", {}).get("f1", 0.0)),
        title="OLMoE (CNN/DM) ROUGE-L & Speedup vs Expert Cap",
        output=Path("figures/pruning_strategies_sum.png"),
    ),
    "alpaca": DatasetConfig(
        metric_label="F1",
        metric_fn=lambda data: float(data.get("f1", 0.0)),
        title="OLMoE (Alpaca) F1 & Speedup vs Expert Cap",
        output=Path("figures/pruning_strategies_alpaca.png"),
    ),
    "humaneval": DatasetConfig(
        metric_label="pass@1",
        metric_fn=lambda data: float(data.get("pass@1", 0.0)),
        title="OLMoE (HumanEval) pass@1 & Speedup vs Expert Cap",
        output=Path("figures/pruning_strategies_humaneval.png"),
    ),
}


def canonical_temperature(value) -> str:
    """Convert temperature value to canonical string."""
    if value is None:
        return "1.0"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{val:.1f}"


def temp_sort_key(label: str) -> float:
    """Sort key for temperature labels."""
    try:
        return float(label)
    except (TypeError, ValueError):
        return float("inf")


def parse_model_prefix(model_id: str) -> Optional[str]:
    """Extract model shortcut prefix from a run directory name.

    Expected format: <model_prefix>-t{temp}-...
    """
    idx = model_id.find("-t")
    if idx == -1:
        return None
    prefix = model_id[:idx]
    return prefix or None


def discover_model_prefixes(dataset: str, root: Path) -> List[str]:
    """List unique model prefixes present for a dataset."""
    results_dir = root / "results" / dataset
    if not results_dir.exists():
        return []

    prefixes: Set[str] = set()
    for metrics_path in results_dir.glob("*/metrics.json"):
        model_id = metrics_path.parent.name
        prefix = parse_model_prefix(model_id)
        if prefix:
            prefixes.add(prefix)
    return sorted(prefixes)


def load_runs(
    dataset: str,
    root: Path,
    draft_tokens: int = 63,
    model_prefix: Optional[str] = None,
) -> Dict[str, Dict[str, Tuple[List[Tuple[float, float, float, float]], Dict[str, Tuple[float, float]]]]]:
    """Load run data grouped by temperature and strategy.

    Args:
        dataset: Dataset name (e.g., "humaneval")
        root: Repository root path
        draft_tokens: Filter for specific draft token count (default: 63)
        model_prefix: Optional model shortcut prefix (e.g., "olmoe1b7b") to filter runs.

    Returns:
        Dict[temp_key, Dict[strategy_key, (points, references)]]
        where strategy_key = "cardinality-substitution", "probability-truncation", etc.
        points = [(effective_cap, metric, throughput, accept_length), ...]
        effective_cap is the actual expert count (int for cardinality, mean for probability)
        and references = {mode: (metric, throughput), ...}
    """
    # Structure: temp -> strategy_key -> data
    temp_map: Dict[str, Dict[str, List[Tuple[float, float, float, float]]]] = {}
    references: Dict[str, Dict[str, Tuple[float, float]]] = {}

    # Pattern: results/{dataset}/olmoe1b7b-t{temp}-eagle3-draft{tokens}-cap{N}-{strategy}/metrics.json
    results_dir = root / "results" / dataset
    if not results_dir.exists():
        return {}

    for metrics_path in results_dir.glob("*/metrics.json"):
        model_id = metrics_path.parent.name

        if model_prefix and not (model_id == model_prefix or model_id.startswith(f"{model_prefix}-")):
            continue

        # Load metrics and summary
        metrics_data = json.loads(metrics_path.read_text())
        summary_path = metrics_path.parent / "summary.json"
        if not summary_path.exists():
            continue
        summary_data = json.loads(summary_path.read_text())

        metric_val = DATASETS[dataset].metric_fn(metrics_data)
        throughput = float(summary_data.get("mean_throughput", 0.0))
        accept_length = float(summary_data.get("mean_accept_length", 0.0))

        # Parse model ID to extract parameters
        # Format: olmoe1b7b-t{temp}-{mode}-...
        parts = model_id.split("-")

        # Extract temperature (must be second element after model name)
        temp_key = None
        for i, part in enumerate(parts):
            if part.startswith("t") and len(part) > 1 and i == 1:  # Temperature is at index 1
                temp_raw = part[1:].replace("p", ".")
                try:
                    float(temp_raw)  # Validate it's a number
                    temp_key = canonical_temperature(temp_raw)
                    break
                except ValueError:
                    pass

        if temp_key is None:
            continue

        # Check if autoregressive baseline
        if "autoregressive" in model_id:
            references.setdefault(temp_key, {})["AR baseline"] = (metric_val, throughput)
            continue

        # Extract draft tokens, cap/probability budget, and strategy
        model_draft_tokens = None
        cap_value = None
        probability_budget = None
        effective_cap = None
        strategy = None
        selection_mode = None  # "cardinality" or "probability"

        for i, part in enumerate(parts):
            if part.startswith("draft") and len(part) > 5:
                try:
                    model_draft_tokens = int(part[5:])
                except ValueError:
                    pass
            elif part == "nocap":
                # EAGLE uncapped run - only include if draft tokens match
                if model_draft_tokens == draft_tokens:
                    references.setdefault(temp_key, {})["EAGLE3"] = (metric_val, throughput)
                break
            elif part.startswith("cap") and len(part) > 3:
                try:
                    cap_value = int(part[3:])
                    selection_mode = "cardinality"
                except ValueError:
                    pass
            elif part.startswith("p") and len(part) > 1 and part[1:].isdigit():
                # Probability budget (e.g., p60 means 0.60)
                try:
                    probability_budget = int(part[1:]) / 100.0
                    selection_mode = "probability"
                except ValueError:
                    pass
            elif part in ["substitution", "truncation"]:
                strategy = part

        # If we found nocap, skip to next file
        if "nocap" in parts:
            continue

        # Only include runs with matching draft tokens
        if model_draft_tokens != draft_tokens or strategy is None:
            continue

        # For cardinality mode, use cap_value directly
        if selection_mode == "cardinality" and cap_value is not None:
            effective_cap = float(cap_value)
            strategy_key = f"cardinality-{strategy}"
        # For probability mode, use mean_active_experts from summary
        elif selection_mode == "probability" and probability_budget is not None:
            mean_active = summary_data.get("mean_active_experts")
            if mean_active is None or mean_active == 0:
                continue  # Skip if data not available
            effective_cap = float(mean_active)
            strategy_key = f"probability-{strategy}"
        else:
            continue

        temp_map.setdefault(temp_key, {}).setdefault(strategy_key, []).append(
            (effective_cap, metric_val, throughput, accept_length)
        )

    # Group and sort
    grouped: Dict[str, Dict[str, Tuple[List[Tuple[float, float, float, float]], Dict[str, Tuple[float, float]]]]] = {}
    for temp_key, strategy_data in temp_map.items():
        grouped[temp_key] = {}
        for strategy_key, points in strategy_data.items():
            sorted_points = sorted(points, key=lambda x: x[0])
            grouped[temp_key][strategy_key] = (sorted_points, references.get(temp_key, {}))

    return grouped


def plot_dataset(
    config: DatasetConfig,
    dataset_name: str,
    temp_key: str,
    strategy_data: Dict[str, Tuple[List[Tuple[float, float, float, float]], Dict[str, Tuple[float, float]]]],
    output_path: Path,
) -> None:
    """Plot comparison of cardinality vs probability and substitution vs truncation strategies."""
    # Get baseline reference (should be same for all strategies)
    references = None
    for _, (_, refs) in strategy_data.items():
        if refs:
            references = refs
            break

    if references is None:
        raise ValueError(f"No references found for {dataset_name} @ temp={temp_key}")

    baseline = references.get("AR baseline")
    if baseline is None or baseline[0] == 0 or baseline[1] == 0:
        raise ValueError(f"Missing or invalid AR baseline for {dataset_name} @ temp={temp_key}")

    fig, (ax_metric, ax_speed) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    # Color by pruning strategy, linestyle by selection mode
    colors = {"substitution": "tab:blue", "truncation": "tab:orange"}
    markers = {"substitution": "o", "truncation": "s"}
    linestyles = {"cardinality": "-", "probability": "--"}

    # Plot all 4 curves: cardinality-{substitution,truncation}, probability-{substitution,truncation}
    for selection_mode in ["cardinality", "probability"]:
        for pruning_strategy in ["substitution", "truncation"]:
            strategy_key = f"{selection_mode}-{pruning_strategy}"
            if strategy_key not in strategy_data:
                continue

            points, _ = strategy_data[strategy_key]
            if not points:
                continue

            caps = [cap for cap, _, _, _ in points]
            rel_metric = [metric / baseline[0] for _, metric, _, _ in points]
            speedup = [thr / baseline[1] for _, _, thr, _ in points]
            accept_lengths = [accept for _, _, _, accept in points]

            # Create label that shows both selection mode and pruning strategy
            label = f"{pruning_strategy.capitalize()} ({selection_mode})"

            ax_metric.plot(
                caps, rel_metric,
                marker=markers[pruning_strategy],
                color=colors[pruning_strategy],
                linestyle=linestyles[selection_mode],
                linewidth=2,
                label=label,
                markersize=8
            )
            ax_speed.plot(
                caps, speedup,
                marker=markers[pruning_strategy],
                color=colors[pruning_strategy],
                linestyle=linestyles[selection_mode],
                linewidth=2,
                label=label,
                markersize=8
            )

            # Add acceptance rate annotations below speedup points (only for cardinality to avoid clutter)
            if selection_mode == "cardinality":
                for x, y, accept in zip(caps, speedup, accept_lengths):
                    if accept > 0:
                        ax_speed.annotate(
                            f'{accept:.1f}',
                            xy=(x, y),
                            xytext=(0, -12),
                            textcoords='offset points',
                            ha='center',
                            va='top',
                            fontsize=8,
                            color=colors[pruning_strategy],
                            alpha=0.7
                        )

    # Add EAGLE3 uncapped reference if available
    uncapped = references.get("EAGLE3")
    if uncapped:
        # Find max cap across both strategies
        max_cap = 0
        for strategy_points, _ in strategy_data.values():
            if strategy_points:
                max_cap = max(max_cap, max(cap for cap, _, _, _ in strategy_points))

        rel_metric = uncapped[0] / baseline[0]
        speedup = uncapped[1] / baseline[1]
        marker_x = max_cap + 4
        ax_metric.scatter([marker_x], [rel_metric], marker="D", s=100, color="tab:purple", label="EAGLE3 (uncapped)", zorder=5)
        ax_speed.scatter([marker_x], [speedup], marker="D", s=100, color="tab:purple", label="EAGLE3 (uncapped)", zorder=5)

    # Formatting
    ax_metric.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_metric.set_ylabel(f"Relative {config.metric_label}\n(vs AR baseline)", fontsize=11)
    ax_metric.grid(True, alpha=0.3)
    ax_metric.legend(loc="best", framealpha=0.9, fontsize=10)
    ax_metric.tick_params(labelbottom=False)

    ax_speed.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_speed.set_xlabel("Average Number of Experts", fontsize=11)
    ax_speed.set_ylabel("Speedup\n(vs AR baseline)", fontsize=11)
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(loc="best", framealpha=0.9, fontsize=9)

    fig.suptitle(f"{config.title}\n(temperature={temp_key})", fontsize=14, fontweight='bold')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot to {output_path} (temp={temp_key})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pruning strategy comparison for OLMoE.")
    parser.add_argument("--results-root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS.keys()),
        help="Comma-separated dataset names (default: all).",
    )
    parser.add_argument(
        "--draft-tokens",
        type=int,
        default=63,
        help="Filter for specific draft token count (default: 63).",
    )
    args = parser.parse_args()

    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not selected:
        selected = list(DATASETS.keys())
    for name in selected:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset '{name}'")

    for dataset_name in selected:
        cfg = DATASETS[dataset_name]
        prefixes = discover_model_prefixes(dataset_name, args.results_root)
        if not prefixes:
            print(f"Skipping {dataset_name}: no sweep data found for draft_tokens={args.draft_tokens}.")
            continue

        for model_prefix in prefixes:
            grouped = load_runs(dataset_name, args.results_root, args.draft_tokens, model_prefix)
            if not grouped:
                print(
                    f"Skipping {dataset_name} ({model_prefix}): no sweep data found for draft_tokens={args.draft_tokens}."
                )
                continue

            for temp_key in sorted(grouped.keys(), key=temp_sort_key):
                strategy_data = grouped[temp_key]
                output_name = (
                    f"{model_prefix}_{cfg.output.stem}_draft{args.draft_tokens}_temp{temp_key.replace('.', 'p')}"
                    f"{cfg.output.suffix}"
                )
                output_path = cfg.output.with_name(output_name)
                try:
                    plot_dataset(cfg, dataset_name, temp_key, strategy_data, output_path)
                except ValueError as exc:
                    print(f"[warn] {exc}")


if __name__ == "__main__":
    main()
