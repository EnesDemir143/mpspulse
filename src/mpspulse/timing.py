"""MPS event timing port of NVIDIA `benchmark_detailed` (needs torch).

Deviation from NVIDIA: the direct port (2n pre-allocated event pairs, one
final sync, bulk read) fails intermittently on MPS (~15%: "End event was
not recorded after start event"). A single reused pair with per-iteration
sync + immediate read is reliable (0 failures in testing). The per-iter
sync sits after `end.record()`, so sync overhead is NOT included in the
measured GPU time — it only forces event timestamp ordering.
"""

from __future__ import annotations

from collections.abc import Callable

from mpspulse.errors import MpsUnavailableError
from mpspulse.stats import summarize


def _require_mps() -> None:
    try:
        import torch
    except ImportError as exc:
        raise MpsUnavailableError(
            "torch is not installed; install the 'mps' extra: uv sync --extra mps"
        ) from exc
    if not torch.backends.mps.is_available():
        raise MpsUnavailableError(
            "torch.backends.mps.is_available() is False; Apple Silicon MPS required"
        )


def benchmark_fn(
    fn: Callable[[], None],
    *,
    warmup: int = 50,
    iters: int = 100,
    device: str = "mps",
) -> dict:
    """Time `fn` per-iteration with a reused MPS event pair.

    Warmup excluded; returns mean/median/std/min/max/n in ms.
    Raises ValueError if iters<1, MpsUnavailableError if MPS unavailable.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if device != "mps":
        raise MpsUnavailableError(f"only device='mps' supported, got {device!r}")
    _require_mps()

    import torch

    for _ in range(warmup):
        fn()
    torch.mps.synchronize()

    start = torch.mps.event.Event(enable_timing=True)
    end = torch.mps.event.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.mps.synchronize()
        times.append(start.elapsed_time(end))
    return summarize(times)
