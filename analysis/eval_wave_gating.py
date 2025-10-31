#!/usr/bin/env python3
"""Evaluate wave-level gating trade-offs using a fitted wave survival model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd

# Ensure repository root on sys.path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from analysis.train_wave_survival_model import WAVE_FEATURES, build_wave_dataframe


def parse_float_list(raw: str | None, default: Iterable[float]) -> List[float]:
    if raw is None:
        return list(default)
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values or list(default)


def load_wave_data(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    wave_df = build_wave_dataframe(df)
    if "wave_total_unique" not in wave_df.columns:
        raise ValueError("wave dataframe missing 'wave_total_unique'; ensure traces include unique expert stats.")
    wave_df["wave_cost"] = wave_df["wave_total_unique"]
    wave_df["wave_accept_frac"] = (
        wave_df["accepted_count"] / wave_df["nodes"] if "nodes" in wave_df else 0.0
    )
    return wave_df


def apply_gating(
    wave_df: pd.DataFrame,
    preds: np.ndarray,
    cost_quantile: float,
    pred_quantile: float,
) -> pd.Series:
    cost_threshold = float(np.quantile(wave_df["wave_cost"], cost_quantile))
    pred_threshold = float(np.quantile(preds, pred_quantile))
    mask = np.ones(len(wave_df), dtype=bool)
    mask &= wave_df["wave_cost"].to_numpy() <= cost_threshold
    mask &= preds <= pred_threshold
    return pd.Series(mask, index=wave_df.index)


def evaluate_gating(
    wave_df: pd.DataFrame,
    mask: pd.Series,
) -> dict[str, float]:
    kept = wave_df[mask]
    pruned = wave_df[~mask]

    total_cost = wave_df["wave_cost"].sum()
    kept_cost = kept["wave_cost"].sum()
    pruned_cost = pruned["wave_cost"].sum()

    total_accepts = wave_df["accepted_count"].sum()
    kept_accepts = kept["accepted_count"].sum()
    pruned_accepts = pruned["accepted_count"].sum()

    stats = {
        "waves_total": len(wave_df),
        "waves_kept": len(kept),
        "waves_kept_frac": len(kept) / len(wave_df) if len(wave_df) else 0.0,
        "cost_total": total_cost,
        "cost_kept": kept_cost,
        "cost_kept_frac": kept_cost / total_cost if total_cost else 0.0,
        "accepted_total": total_accepts,
        "accepted_kept": kept_accepts,
        "accepted_kept_frac": kept_accepts / total_accepts if total_accepts else 0.0,
        "accepted_pruned_frac": pruned_accepts / total_accepts if total_accepts else 0.0,
        "cost_pruned_frac": pruned_cost / total_cost if total_cost else 0.0,
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Training trace Parquet file (with wave-level schema).",
    )
    parser.add_argument(
        "--wave-model",
        type=Path,
        required=True,
        help="Fitted wave survival model (joblib).",
    )
    parser.add_argument(
        "--cost-quantiles",
        type=str,
        default=None,
        help="Comma-separated cost quantiles (default: 0.6,0.7,0.8,0.9)",
    )
    parser.add_argument(
        "--pred-quantiles",
        type=str,
        default=None,
        help="Comma-separated prediction quantiles (default: 0.6,0.7,0.8,0.9)",
    )
    parser.add_argument(
        "--utility-lambdas",
        type=str,
        default=None,
        help="Comma-separated lambda values for utility gating (predict - lambda * cost > 0)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/combined/wave_gating_tradeoff.csv"),
        help="Output CSV summarizing the sweep.",
    )
    parser.add_argument(
        "--include-oracle",
        action="store_true",
        help="When using utility lambdas, also compute the oracle curve with ground-truth accepted counts.",
    )
    args = parser.parse_args()

    utility_lambdas = parse_float_list(args.utility_lambdas, default=[]) if args.utility_lambdas else []
    cost_quantiles = parse_float_list(args.cost_quantiles, default=[0.6, 0.7, 0.8, 0.9])
    pred_quantiles = parse_float_list(args.pred_quantiles, default=[0.6, 0.7, 0.8, 0.9])

    wave_df = load_wave_data(args.data)
    model = joblib.load(args.wave_model)
    preds = model.predict(wave_df[WAVE_FEATURES])

    rows = []
    baseline_stats = evaluate_gating(wave_df, pd.Series(True, index=wave_df.index))
    baseline_stats.update(
        {
            "cost_quantile": np.nan,
            "pred_quantile": np.nan,
            "lambda": np.nan,
            "mode": "baseline",
        }
    )
    rows.append(baseline_stats)

    if utility_lambdas:
        cost_array = wave_df["wave_cost"].to_numpy()
        for lam in utility_lambdas:
            utility_pred = preds - lam * cost_array
            mask_pred = utility_pred > 0
            stats_pred = evaluate_gating(wave_df, pd.Series(mask_pred, index=wave_df.index))
            stats_pred.update(
                {
                    "cost_quantile": np.nan,
                    "pred_quantile": np.nan,
                    "lambda": lam,
                    "mode": "predicted",
                }
            )
            rows.append(stats_pred)

            if args.include_oracle:
                oracle_util = wave_df["accepted_count"].to_numpy() - lam * cost_array
                mask_oracle = oracle_util > 0
                stats_oracle = evaluate_gating(wave_df, pd.Series(mask_oracle, index=wave_df.index))
                stats_oracle.update(
                    {
                        "cost_quantile": np.nan,
                        "pred_quantile": np.nan,
                        "lambda": lam,
                        "mode": "oracle",
                    }
                )
                rows.append(stats_oracle)
    else:
        for cq in cost_quantiles:
            for pq in pred_quantiles:
                mask = apply_gating(wave_df, preds, cost_quantile=cq, pred_quantile=pq)
                stats = evaluate_gating(wave_df, mask)
                stats.update({
                    "cost_quantile": cq,
                    "pred_quantile": pq,
                    "lambda": np.nan,
                    "mode": "predicted",
                })
                rows.append(stats)

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved sweep to {args.out}")


if __name__ == "__main__":
    main()
