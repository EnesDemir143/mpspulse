"""Training-time ETA engine (torch-free core).

Interval estimates from measured step-time samples — never a single
number. Sources: `static` (caller-supplied, low confidence), `benchmark`
(short live measurement), `live` (incremental updates from a real loop).
Calibration history lives in ~/.mpspulse/history.jsonl.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EtaReport:
    p50_s: float
    p10_s: float
    p90_s: float
    n_samples: int
    source: str  # static | benchmark | live
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "p50_s": self.p50_s,
            "p10_s": self.p10_s,
            "p90_s": self.p90_s,
            "n_samples": self.n_samples,
            "source": self.source,
            "warnings": self.warnings,
        }


def _quantile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def estimate(
    total_steps: int,
    step_ms_samples: list[float],
    *,
    startup_s: float = 0.0,
    val_overhead_per_epoch_s: float = 0.0,
    ckpt_overhead_per_epoch_s: float = 0.0,
    epochs: int = 1,
    grad_accum: int = 1,
    source: str = "static",
) -> EtaReport:
    """Scale step-time samples to a total-time interval.

    `total_steps` counts optimizer steps; with grad_accum>1 each optimizer
    step costs grad_accum micro-steps. Epoch overheads are added as fixed
    costs (no interval) on top of the scaled step interval.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if not step_ms_samples:
        raise ValueError("need at least 1 step-ms sample")
    if grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if source not in ("static", "benchmark", "live"):
        raise ValueError(f"unknown source {source!r}")

    micro_steps = total_steps * grad_accum
    p10 = _quantile(step_ms_samples, 0.1) / 1000 * micro_steps
    p50 = _quantile(step_ms_samples, 0.5) / 1000 * micro_steps
    p90 = _quantile(step_ms_samples, 0.9) / 1000 * micro_steps
    fixed = startup_s + epochs * (val_overhead_per_epoch_s + ckpt_overhead_per_epoch_s)

    warnings: list[str] = []
    if source == "static":
        warnings.append("low-confidence: static step time, not measured on this machine")
    if len(step_ms_samples) < 10:
        warnings.append(f"few samples (n={len(step_ms_samples)}); interval is rough")
    return EtaReport(
        p50_s=p50 + fixed,
        p10_s=p10 + fixed,
        p90_s=p90 + fixed,
        n_samples=len(step_ms_samples),
        source=source,
        warnings=warnings,
    )


class EtaTracker:
    """Live ETA: feed real step times, recompute the interval incrementally."""

    def __init__(self, total_steps: int, **kwargs: Any):
        self.total_steps = total_steps
        self.kwargs = kwargs
        self.samples: list[float] = []

    def update(self, step_ms: float) -> EtaReport:
        self.samples.append(step_ms)
        return estimate(self.total_steps, list(self.samples), source="live", **self.kwargs)


def history_path() -> Path:
    return Path(os.environ.get("MPSPULSE_HOME", str(Path.home()))) / ".mpspulse" / "history.jsonl"


def record_history(model_id: str, config_hash: str, step_ms_p50: float, n: int) -> None:
    path = history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(
            json.dumps(
                {
                    "model_id": model_id,
                    "config_hash": config_hash,
                    "step_ms_p50": step_ms_p50,
                    "n": n,
                }
            )
            + "\n"
        )


def lookup_history(model_id: str, config_hash: str) -> list[dict[str, Any]]:
    path = history_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model_id") == model_id and row.get("config_hash") == config_hash:
            rows.append(row)
    return rows


def drift_warning(model_id: str, config_hash: str, step_ms_p50: float) -> str | None:
    """Warn if current p50 differs >20% from the median of past same-config runs."""
    rows = lookup_history(model_id, config_hash)
    if not rows:
        return None
    past = statistics.median(r["step_ms_p50"] for r in rows)
    if past <= 0:
        return None
    drift = (step_ms_p50 - past) / past
    if abs(drift) > 0.20:
        return (
            f"history drift: current p50 {step_ms_p50:.2f}ms vs past median "
            f"{past:.2f}ms ({drift:+.0%}, n={len(rows)} past runs)"
        )
    return None
