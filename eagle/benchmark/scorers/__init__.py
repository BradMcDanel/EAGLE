"""Dataset-specific scoring helpers."""

from __future__ import annotations

from typing import Any, Dict

from . import gsm8k, sum_scorer


SCORERS = {
    "gsm8k": gsm8k.score,
    "sum": sum_scorer.score,
}


def get_scorer(bench_name: str):
    """Return a scoring callable for the given benchmark, if available."""

    return SCORERS.get(bench_name)

