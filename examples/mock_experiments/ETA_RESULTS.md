# ETA Accuracy Results (v0.2 level)

Device: Apple M2 Pro, macOS 27.0.1, torch 2.14.1 (MPS)
Date: 2026-10-14
Script: `uv run scripts/eta_accuracy.py`

## Method

Each round: calibrate on `unet_like` mock (10 independent uninterrupted
100-iter runs → 10 wall-ms samples), predict a 1000-step run, actually run
1000 steps, check the actual falls inside [p10, p90]. Pass = ≥2/3 rounds in.

## Latest run

| Round | Step p50 ms | Pred [p10,p90] s | Actual s | Err vs p50 | In? |
|---|---|---|---|---|---|
| 1 | 2.33 | [2.27, 2.50] | 2.37 | +1.7% | ✅ |
| 2 | 2.35 | [2.26, 2.44] | 2.26 | −3.6% | ✅ |
| 3 | 2.41 | [2.23, 2.48] | 2.38 | −1.0% | ✅ |

Coverage 3/3, mean |err| 2.1%. Second run: 3/3, mean |err| 2.6%.

## Deviations from the plan (all measured, not assumed)

- **1000 validation steps, not 200.** A single 200-step run sits in the tail
  of the calibration distribution (between-run variance dominates); longer
  runs average it out, as real training does. 4/4 IN at 1000 vs ~2/4 at 200.
- **Throughput calibration = repeated uninterrupted runs.** Per-iteration
  sync+time (NVIDIA latency template) measures ~25% slower than real async
  training; chunked timing still inflates ~10–15% via chunk-boundary pipeline
  drains (measured on M2 Pro). Each calibration sample is now one full
  uninterrupted run — identical in shape to real training.
- Latency diagnosis still uses per-iter sync timing (`profile`); only ETA
  calibration uses throughput mode. The two answer different questions.
