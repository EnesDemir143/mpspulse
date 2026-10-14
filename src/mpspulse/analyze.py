"""Bottleneck classification over measured profile reports (torch-free).

Port of NVIDIA perf-analysis: classify-before-recommending, every finding
traces to measured fields. No MFU/SOL (no Apple peak-FLOP API — omitted,
never invented), no specialist delegation (single-machine local analysis).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DATA_RATIO_THRESHOLD = 0.2
GPU_WALL_THRESHOLD = 0.5
MEMORY_PRESSURE_THRESHOLD = 0.85
CV_THRESHOLD = 0.25


@dataclass
class Finding:
    class_: str
    metric: str
    value: float
    threshold: float
    recommendation: str
    priority: int  # 1 = highest

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.class_,
            "evidence": {"metric": self.metric, "value": self.value, "threshold": self.threshold},
            "recommendation": self.recommendation,
            "priority": self.priority,
        }


def classify(report: dict[str, Any]) -> list[Finding]:
    """Classify bottlenecks from a schema.py profile report.

    Operates ONLY on measured fields; evidence values are copied from input.
    """
    findings: list[Finding] = []
    wall = report["wall_ms"]["mean_ms"]
    data = report["data_ms"]["mean_ms"]
    mem = report["memory"]
    data_ratio = data / wall if wall else 0.0

    if data_ratio > DATA_RATIO_THRESHOLD:
        findings.append(
            Finding(
                "data-loading",
                "data_ms/wall_ms",
                data_ratio,
                DATA_RATIO_THRESHOLD,
                "Profile the dataloader: add workers, enable prefetching/pinning, "
                "or move augmentation to the GPU. Re-measure data_ms after changes.",
                1,
            )
        )

    gpu = report.get("gpu_event_ms")
    if gpu is not None:
        gpu_ratio = gpu["mean_ms"] / wall if wall else 0.0
        if gpu_ratio < GPU_WALL_THRESHOLD and data_ratio <= DATA_RATIO_THRESHOLD:
            findings.append(
                Finding(
                    "host-overhead",
                    "gpu_event_ms/wall_ms",
                    gpu_ratio,
                    GPU_WALL_THRESHOLD,
                    "GPU is idle most of the iteration while data loading is fine: "
                    "suspect Python overhead or host-device syncs. Run "
                    "`mpspulse sync-scan` on the training script.",
                    1,
                )
            )

    rec_max = mem.get("recommended_max", 0)
    if rec_max:
        mem_ratio = mem.get("driver_allocated", 0) / rec_max
        if mem_ratio > MEMORY_PRESSURE_THRESHOLD:
            findings.append(
                Finding(
                    "memory-pressure",
                    "driver_allocated/recommended_max",
                    mem_ratio,
                    MEMORY_PRESSURE_THRESHOLD,
                    "Near the recommended MPS memory limit: reduce batch size, "
                    "enable gradient checkpointing, or clear unneeded caches.",
                    2,
                )
            )

    w = report["wall_ms"]
    if w.get("n", 0) > 1 and w["mean_ms"]:
        cv = w["std_ms"] / w["mean_ms"]
        if cv > CV_THRESHOLD:
            findings.append(
                Finding(
                    "variance",
                    "wall_ms CV",
                    cv,
                    CV_THRESHOLD,
                    "High run-to-run variance: re-run on an idle machine, check "
                    "for variable input shapes, thermal throttling, or background load.",
                    3,
                )
            )

    return sorted(findings, key=lambda f: f.priority)


@dataclass
class AnalysisReport:
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    deep: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "metrics": self.metrics,
            "findings": [f.to_dict() for f in self.findings],
            "deep": self.deep,
        }


def analyze(report: dict[str, Any]) -> AnalysisReport:
    """Build the Summary/Metrics/Findings analysis for a profile report."""
    findings = classify(report)
    wall = report["wall_ms"]
    gpu = report.get("gpu_event_ms")
    metrics = {
        "wall_mean_ms": wall["mean_ms"],
        "wall_std_ms": wall["std_ms"],
        "samples_per_sec": report.get("samples_per_sec", 0.0),
        "data_ratio": report["data_ms"]["mean_ms"] / wall["mean_ms"] if wall["mean_ms"] else 0.0,
        "gpu_wall_ratio": (gpu["mean_ms"] / wall["mean_ms"] if gpu and wall["mean_ms"] else None),
    }
    if not findings:
        summary = "Clean bill: no bottleneck detected in measured fields."
    else:
        top = findings[0]
        summary = (
            f"Primary bottleneck: {top.class_} "
            f"({top.metric}={top.value:.3f} vs threshold {top.threshold})."
        )
    return AnalysisReport(summary=summary, metrics=metrics, findings=findings)
