"""Training-loop profiling port of NVIDIA's injection template (needs torch)."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from mpspulse.stats import summarize
from mpspulse.timing import _require_mps


@dataclass
class LoopReport:
    model: str
    batch_size: int
    iters: int
    warmup: int
    wall_samples_ms: list[float] = field(default_factory=list)
    data_samples_ms: list[float] = field(default_factory=list)
    gpu_event_samples_ms: list[float] | None = None
    phase_samples_ms: dict[str, list[float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def samples_per_sec(self) -> float:
        if not self.wall_samples_ms:
            return 0.0
        mean_s = sum(self.wall_samples_ms) / len(self.wall_samples_ms) / 1000
        return self.batch_size / mean_s if mean_s else 0.0


def profile_loop(
    dataloader: Iterator,
    step_fn: Callable[[object], None],
    *,
    model: str = "custom",
    batch_size: int = 1,
    warmup: int = 5,
    iters: int = 30,
    phases: dict[str, Callable[[object], None]] | None = None,
    gpu_events: bool = True,
    throughput: bool = False,
) -> LoopReport:
    """Profile a training loop: manual sync+perf_counter per iteration.

    Mirrors NVIDIA's injection template: warmup excluded, per-iter wall ms,
    data-load ms measured between iterations, optional per-phase breakdown
    (one-off diagnosis) and per-iter MPS event pairs for GPU-vs-wall split.
    When `phases` is given, `step_fn` is NOT called; the phase callables run
    in order instead, each timed separately (otherwise work would run twice).
    With `throughput=True`, iterations run back-to-back with a single sync
    pair around the whole run (async GPU/CPU overlap preserved, like real
    training); wall samples are per-chunk means over `chunk` iters. Use it
    for ETA calibration, not for latency diagnosis.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    _require_mps()

    import torch

    if throughput:
        return _profile_throughput(
            dataloader, step_fn, model=model, batch_size=batch_size, warmup=warmup, iters=iters
        )

    report = LoopReport(model=model, batch_size=batch_size, iters=iters, warmup=warmup)
    phase_names = list(phases or {})
    phase_acc: dict[str, list[float]] = {name: [] for name in phase_names}

    gpu_start = gpu_end = None
    if gpu_events:
        gpu_start = torch.mps.event.Event(enable_timing=True)
        gpu_end = torch.mps.event.Event(enable_timing=True)
        report.gpu_event_samples_ms = []

    prev_iter_end: float | None = None
    collected = 0
    total = warmup + iters
    for i, batch in enumerate(dataloader):
        if i >= total:
            break
        t_data_end = time.perf_counter()

        torch.mps.synchronize()
        t_start = time.perf_counter()

        if i >= warmup and gpu_events:
            assert gpu_start is not None
            gpu_start.record()
        if phase_names:
            assert phases is not None
            for name in phase_names:
                torch.mps.synchronize()
                t0 = time.perf_counter()
                phases[name](batch)
                torch.mps.synchronize()
                if i >= warmup:
                    phase_acc[name].append((time.perf_counter() - t0) * 1000)
        else:
            step_fn(batch)
        if i >= warmup and gpu_events:
            assert gpu_end is not None and gpu_start is not None
            gpu_end.record()

        torch.mps.synchronize()
        t_end = time.perf_counter()
        if i >= warmup and gpu_events:
            assert report.gpu_event_samples_ms is not None
            report.gpu_event_samples_ms.append(gpu_start.elapsed_time(gpu_end))

        if i >= warmup:
            report.wall_samples_ms.append((t_end - t_start) * 1000)
            if prev_iter_end is not None:
                report.data_samples_ms.append((t_data_end - prev_iter_end) * 1000)
            collected += 1
        prev_iter_end = t_end

    if collected < iters:
        report.warnings.append(f"dataloader exhausted after {collected}/{iters} measured iters")
        report.iters = collected
    report.phase_samples_ms = phase_acc
    return report


def _profile_throughput(
    dataloader, step_fn, *, model, batch_size, warmup, iters, chunk: int = 10
) -> LoopReport:
    """Back-to-back timing: repeats of one uninterrupted chunk per sample.

    Each wall sample is a full `iters`-long run bracketed by a single sync
    pair — identical in shape to a real training run, so chunk-boundary
    pipeline drains (measured ~10-15% on M2 Pro) never inflate calibration.
    `chunk` sets the repeat count (default 10 samples).
    """
    import torch

    report = LoopReport(model=model, batch_size=batch_size, iters=iters, warmup=warmup)
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= warmup + iters:
            break
        batches.append(batch)
    if len(batches) < warmup + iters:
        report.warnings.append(
            f"dataloader exhausted: {len(batches)}/{warmup + iters} iters available"
        )
    warm_batches = batches[:warmup] or [None]
    timed_batches = batches[warmup : warmup + iters] or [None]
    for _ in range(chunk):
        for batch in warm_batches:
            step_fn(batch)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for batch in timed_batches:
            step_fn(batch)
        torch.mps.synchronize()
        report.wall_samples_ms.append((time.perf_counter() - t0) * 1000 / len(timed_batches))
    report.warnings.append(
        "throughput mode: each sample is one uninterrupted run (async overlap kept)"
    )
    return report


def loop_report_to_dict(report: LoopReport, *, device: str, memory: dict) -> dict:
    """Convert a LoopReport to the schema.py ProfileReport JSON shape."""
    from mpspulse.schema import SCHEMA_VERSION, TOOL

    data = report.data_samples_ms or [0.0]
    gpu = None
    if report.gpu_event_samples_ms:
        gpu = summarize(report.gpu_event_samples_ms)
    return {
        "tool": TOOL,
        "version": SCHEMA_VERSION,
        "device": device,
        "model": report.model,
        "batch_size": report.batch_size,
        "iters": report.iters,
        "warmup": report.warmup,
        "gpu_event_ms": gpu,
        "wall_ms": summarize(report.wall_samples_ms or [0.0]),
        "data_ms": summarize(data),
        "samples_per_sec": report.samples_per_sec,
        "memory": memory,
        "warnings": report.warnings,
    }
