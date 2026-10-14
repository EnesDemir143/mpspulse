"""MPS-gated timing tests (skipped without torch/MPS)."""

import statistics

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.mps

if not torch.backends.mps.is_available():
    pytest.skip("MPS unavailable", allow_module_level=True)

from mpspulse.timing import benchmark_fn


def _matmul(n=1024):
    x = torch.randn(n, n, device="mps")
    return lambda: x @ x


def test_event_vs_wall_sanity():
    """GPU event time must not exceed end-to-end wall time."""
    import time

    torch.mps.synchronize()
    t0 = time.perf_counter()
    stats = benchmark_fn(_matmul(), warmup=5, iters=20)
    torch.mps.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000 / 20
    assert stats["mean_ms"] <= wall_ms
    assert stats["n"] == 20


def test_repeatability_cv_under_15pct():
    # First run absorbs one-off machine noise (cold caches, clock ramp);
    # steady-state is judged on the two follow-up runs.
    benchmark_fn(_matmul(), warmup=10, iters=30)
    runs = [benchmark_fn(_matmul(), warmup=10, iters=30)["mean_ms"] for _ in range(2)]
    mean = statistics.mean(runs)
    cv = abs(runs[0] - runs[1]) / mean
    assert cv < 0.15, f"CV {cv:.3f} too high: {runs}"


def test_empty_iters_raises():
    with pytest.raises(ValueError):
        benchmark_fn(_matmul(), iters=0)


def test_memory_keys_nonzero():
    x = torch.randn(128, 128, device="mps")
    y = x @ x
    torch.mps.synchronize()
    assert torch.mps.current_allocated_memory() > 0
    assert torch.mps.driver_allocated_memory() > 0
    assert torch.mps.recommended_max_memory() > 0
    del y
