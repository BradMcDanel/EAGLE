#!/usr/bin/env python3
"""Compute expert efficiency metrics from raw tree trace data."""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Compute expert efficiency metrics")
    parser.add_argument('--waves', required=True, help='Path to waves.csv (from analyze_tree_experts)')
    parser.add_argument('--out', default='efficiency.csv', help='Output CSV path')
    args = parser.parse_args()

    df = pd.read_csv(args.waves)
    if "total_unique_experts" in df:
        total_unique_col = "total_unique_experts"
    elif "total_unique_all" in df:
        total_unique_col = "total_unique_all"
    else:
        raise ValueError("waves.csv missing total_unique_experts/total_unique_all column")

    df['accepted_ratio'] = df['accepted_len'] / df['nodes'].clip(lower=1)
    df['efficiency_score'] = df['accepted_ratio'] / df[total_unique_col].clip(lower=1)

    width_cols = [col for col in df.columns if col.startswith('width_depth_')]
    cols = ['iteration', 'depth', 'accepted_len', total_unique_col, 'efficiency_score'] + width_cols
    df.sort_values('efficiency_score', ascending=False)[cols].to_csv(args.out, index=False)
    print(f"Wrote efficiency table to {args.out}")


if __name__ == '__main__':
    main()
