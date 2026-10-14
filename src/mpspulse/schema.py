"""Profile-report JSON schema (torch-free).

The core contract: `gpu_event_ms` (nullable hardware GPU time) is always
kept separate from `wall_ms` (end-to-end pipeline time). Every consumer
(ETA, analysis, agent checks) depends on this split.
"""

from __future__ import annotations

from typing import Any, TypedDict

TOOL = "mpspulse"
SCHEMA_VERSION = "0.1"

MEMORY_KEYS = ("current_allocated", "driver_allocated", "recommended_max")
SUMMARY_KEYS = ("mean_ms", "median_ms", "std_ms", "min_ms", "max_ms", "n")


class MemoryInfo(TypedDict):
    current_allocated: int
    driver_allocated: int
    recommended_max: int


class TimingSummary(TypedDict):
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n: int


class ProfileReport(TypedDict):
    tool: str
    version: str
    device: str
    model: str
    batch_size: int
    iters: int
    warmup: int
    gpu_event_ms: TimingSummary | None
    wall_ms: TimingSummary
    data_ms: TimingSummary
    samples_per_sec: float
    memory: MemoryInfo
    warnings: list[str]


def validate_report(report: dict[str, Any]) -> list[str]:
    """Return a list of schema violations (empty = valid)."""
    errors: list[str] = []
    for key in (
        "tool",
        "version",
        "device",
        "model",
        "batch_size",
        "iters",
        "warmup",
        "gpu_event_ms",
        "wall_ms",
        "data_ms",
        "samples_per_sec",
        "memory",
        "warnings",
    ):
        if key not in report:
            errors.append(f"missing key: {key}")
    if report.get("tool") != TOOL:
        errors.append(f"tool must be {TOOL!r}")
    for key in ("wall_ms", "data_ms"):
        block = report.get(key)
        if isinstance(block, dict):
            for sub in SUMMARY_KEYS:
                if sub not in block:
                    errors.append(f"missing key: {key}.{sub}")
        elif key in report:
            errors.append(f"{key} must be an object")
    gpu = report.get("gpu_event_ms")
    if gpu is not None:
        if isinstance(gpu, dict):
            for sub in SUMMARY_KEYS:
                if sub not in gpu:
                    errors.append(f"missing key: gpu_event_ms.{sub}")
        else:
            errors.append("gpu_event_ms must be an object or null")
    mem = report.get("memory")
    if isinstance(mem, dict):
        for sub in MEMORY_KEYS:
            if sub not in mem:
                errors.append(f"missing key: memory.{sub}")
    elif "memory" in report:
        errors.append("memory must be an object")
    if "warnings" in report and not isinstance(report["warnings"], list):
        errors.append("warnings must be a list")
    return errors


def dry_run_fixture(model: str = "resnet50", batch_size: int = 8) -> dict[str, Any]:
    """Schema-shaped placeholder emitted by `--dry-run` without torch.

    Values are zeros, never plausible measurements — nothing may mistake
    this for a real benchmark.
    """
    zero = {"mean_ms": 0.0, "median_ms": 0.0, "std_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "n": 0}
    return {
        "tool": TOOL,
        "version": SCHEMA_VERSION,
        "device": "dry-run (no measurement)",
        "model": model,
        "batch_size": batch_size,
        "iters": 0,
        "warmup": 0,
        "gpu_event_ms": None,
        "wall_ms": dict(zero),
        "data_ms": dict(zero),
        "samples_per_sec": 0.0,
        "memory": {"current_allocated": 0, "driver_allocated": 0, "recommended_max": 0},
        "warnings": ["dry-run: no measurement performed"],
    }
