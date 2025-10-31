#!/usr/bin/env python3
"""Correlate MoE layer activation levels across draft/verify cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def load_layer_stats(
    trace_paths: Iterable[Path],
    labels: Iterable[str],
    use_total: bool,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for path, label in zip(trace_paths, labels):
        with path.open() as fh:
            for line in fh:
                record = json.loads(line)
                for choice in record.get("choices", []):
                    for stats in choice.get("stats", []):
                        for wave in stats.get("expert_traces", []):
                            layer_stats = wave.get("layer_expert_stats")
                            if not layer_stats:
                                continue
                            row: Dict[str, float] = {
                                "dataset": label,
                                "iteration": wave.get("iteration", -1),
                                "accepted_len": wave.get("accepted_length", 0),
                            }
                            for layer_key, layer_data in layer_stats.items():
                                metric = "total_unique" if use_total else "accepted_unique"
                                row[f"layer_{layer_key}"] = layer_data.get(metric, 0.0)
                            rows.append(row)
    return pd.DataFrame(rows)


def compute_correlations(
    df: pd.DataFrame,
    out_dir: Path,
    early_layers: List[int],
    late_layers: List[int],
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_cols = sorted(
        [col for col in df.columns if col.startswith("layer_")],
        key=lambda c: int(c.split("_")[1]),
    )
    available_layers = [int(col.split("_")[1]) for col in layer_cols]

    summaries: List[Dict[str, float]] = []

    for dataset, group in df.groupby("dataset"):
        layer_values = group[layer_cols].copy()
        corr = layer_values.corr()
        corr.to_csv(out_dir / f"{dataset}_layer_corr.csv")

        def sum_layers(indices: Iterable[int]) -> pd.Series:
            cols = [f"layer_{idx}" for idx in indices if idx in available_layers]
            if not cols:
                return pd.Series(np.zeros(len(layer_values)), index=layer_values.index)
            return layer_values[cols].sum(axis=1)

        early = sum_layers(early_layers)
        late = sum_layers(late_layers)
        mid_candidates = [idx for idx in available_layers if idx not in set(early_layers) | set(late_layers)]
        mid = sum_layers(mid_candidates)

        def safe_corr(a: pd.Series, b: pd.Series) -> float:
            if a.std() == 0 or b.std() == 0:
                return float("nan")
            return float(a.corr(b))

        summaries.append(
            {
                "dataset": dataset,
                "corr_early_mid": safe_corr(early, mid),
                "corr_early_late": safe_corr(early, late),
                "corr_mid_late": safe_corr(mid, late),
            }
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "layer_block_correlations.csv", index=False)
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlate layer activations")
    parser.add_argument("trace_files", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument(
        "--use-total",
        action="store_true",
        help="Use total_unique (whole tree) instead of accepted_unique",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/combined/layer_correlations"),
    )
    parser.add_argument(
        "--early",
        nargs="*",
        type=int,
        default=[0, 1, 2],
        help="Layer indices treated as early",
    )
    parser.add_argument(
        "--late",
        nargs="*",
        type=int,
        default=[12, 13, 14, 15],
        help="Layer indices treated as late",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    labels = args.labels or [path.parent.name for path in args.trace_files]
    if len(labels) != len(args.trace_files):
        raise ValueError("Number of labels must match number of trace files")

    df = load_layer_stats(args.trace_files, labels, use_total=args.use_total)
    if df.empty:
        print("No layer stats found in provided traces")
        return

    summary_df = compute_correlations(df, args.out_dir, args.early, args.late)
    print(summary_df)


if __name__ == "__main__":
    main()
