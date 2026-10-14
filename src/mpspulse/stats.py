"""Pure-stdlib sample statistics (torch-free)."""

from __future__ import annotations

import statistics


def summarize(samples: list[float]) -> dict:
    """Mean/median/std/min/max/n for per-iteration timings in ms."""
    if len(samples) < 1:
        raise ValueError("need at least 1 sample")
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "n": len(samples),
    }


def cv(samples: list[float]) -> float:
    """Coefficient of variation (std/mean); 0.0 for degenerate input."""
    if len(samples) < 2:
        return 0.0
    mean = statistics.mean(samples)
    return statistics.stdev(samples) / mean if mean else 0.0
