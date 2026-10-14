"""ETA accuracy check: predict a 1000-step run, actually run it, compare.

Usage: uv run scripts/eta_accuracy.py
Runs 3 calibration+validation rounds. Exit 0 iff at least 2 of 3 actual
totals fall inside their predicted [p10, p90] interval — an honest coverage
test for an 80% interval under real machine noise.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "mock_experiments"))

import torch
import unet_like

from mpspulse.eta import estimate
from mpspulse.loop import profile_loop

STEPS = 1000
ROUNDS = 3


def one_round() -> tuple[bool, float]:
    report = profile_loop(
        iter([None] * 120),
        lambda b: unet_like.train_step(),
        model="unet_like",
        batch_size=4,
        warmup=20,
        iters=100,
        gpu_events=False,
        throughput=True,
    )
    samples = report.wall_samples_ms
    pred = estimate(STEPS, samples, source="benchmark")

    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        unet_like.train_step()
    torch.mps.synchronize()
    actual = time.perf_counter() - t0

    inside = pred.p10_s <= actual <= pred.p90_s
    err_pct = (actual - pred.p50_s) / pred.p50_s * 100
    print(
        f"step p50: {statistics.median(samples):.2f}ms (n={len(samples)}); "
        f"pred=[{pred.p10_s:.2f},{pred.p90_s:.2f}]s actual={actual:.2f}s "
        f"(err {err_pct:+.1f}%) {'IN' if inside else 'OUT'}"
    )
    return inside, err_pct


def main() -> int:
    torch.manual_seed(0)
    results = [one_round() for _ in range(ROUNDS)]
    hits = sum(1 for inside, _ in results if inside)
    mean_err = statistics.mean(abs(e) for _, e in results)
    print(f"coverage: {hits}/{ROUNDS} inside [p10,p90]; mean |err| {mean_err:.1f}%")
    return 0 if hits >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
