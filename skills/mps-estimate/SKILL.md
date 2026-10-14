---
name: mps-estimate
description: >
  Estimate total Apple Silicon PyTorch training time from measured step
  times. Use when the user asks how long training will take, for an ETA, a
  time prediction, or a runtime interval. Runs the mpspulse CLI — reports an
  interval with source and warnings, never a single invented number.
  Triggers: "how long will training take", "estimate training time",
  "ETA for my run", "predict runtime", "time to train".
license: Apache-2.0
compatibility: Requires the mpspulse repo (uv + Python 3.11+); benchmark mode needs macOS on Apple Silicon with torch installed
metadata:
  author: mpspulse contributors
---

# MPS Estimate

Predict total training time from step-time measurements.

## Principles

- **Interval, never a point.** Report [p10, p90] with the p50, sample count,
  source, and warnings.
- **Measured beats guessed.** Prefer `benchmark` (short live measurement) or
  `live` (real step times) over `static` (caller-supplied, low confidence).
- **Know the overheads.** Startup, validation, checkpoint, and gradient
  accumulation change totals — ask when unknown, state assumptions.

## Workflow

1. **Estimate.** From the mpspulse repo:
   ```sh
   # static: caller-supplied step ms (low confidence)
   uv run mpspulse estimate --steps N --epochs E --step-ms X
   # benchmark: short live measurement (preferred)
   uv run mpspulse estimate --steps N --epochs E --mode benchmark \
     --model <builtin|script:mod:fn> --batch-size B
   # live: pipe real step times (one float per line)
   <training loop> | uv run mpspulse estimate --steps N --mode live
   ```
   Add `--grad-accum K --startup-s S --val-s V --ckpt-s C` as known.
2. **Report.** Quote p50 with [p10, p90], source, n_samples, warnings, and
   history-drift notes — from the JSON only.
3. **Calibrate.** Benchmark runs append to `~/.mpspulse/history.jsonl`; a
   >20% drift vs past same-config runs warns automatically.
