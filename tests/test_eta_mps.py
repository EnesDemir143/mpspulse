"""MPS-gated ETA benchmark-mode test."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.mps

if not torch.backends.mps.is_available():
    pytest.skip("MPS unavailable", allow_module_level=True)

import argparse

from mpspulse.cli import _benchmark_wall_samples
from mpspulse.eta import estimate


def test_benchmark_mode_produces_interval():
    args = argparse.Namespace(model="unet_tiny", batch_size=2, iters=10, warmup=3)
    samples = _benchmark_wall_samples(args)
    rep = estimate(500, samples, source="benchmark")
    assert rep.source == "benchmark"
    assert rep.n_samples > 0
    assert rep.p10_s <= rep.p50_s <= rep.p90_s
