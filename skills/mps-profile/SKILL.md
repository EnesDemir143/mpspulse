---
name: mps-profile
description: >
  Profile Apple Silicon PyTorch MPS training steps. Use when the user asks to
  measure throughput, benchmark a training step, time a loop, get
  samples/sec, measure data load time, or compare GPU vs wall-clock time on
  MPS. Runs the mpspulse CLI for real measurements — never invents numbers.
  Triggers: "profile this model", "benchmark my training step",
  "samples per second", "how slow is my loop", "data load time", "MPS timing".
license: Apache-2.0
compatibility: Requires the mpspulse CLI (pip install "mpspulse[mps]"); measurements need macOS on Apple Silicon with torch installed
metadata:
  author: mpspulse contributors
  port_of: NVIDIA perf-workload-profiling
---

# MPS Profile

Measure a representative PyTorch MPS training step with warmup, steady-state
timing, and GPU-vs-wall separation.

## Principles

- **Measure, don't guess.** Every number comes from `mpspulse profile` JSON.
  Never invent metrics.
- **Isolate steady-state.** Warmup iters are excluded before collecting data.
- **Keep GPU and wall separate.** `gpu_event_ms` (hardware GPU time) and
  `wall_ms` (end-to-end) answer different questions — never mix them.
- **Minimal overhead.** Builtin models profile in-process; `script:` targets
  run `--isolated` in a subprocess so caller state is untouched.

## Workflow

1. **Check the CLI.** If `mpspulse --help` fails, tell the user to install it:
   `pip install "mpspulse[mps]"` (measurements need macOS on Apple Silicon).
2. **Profile.** Run:
   ```sh
   mpspulse profile --model <resnet50|vit_b16|unet_tiny|script:mod:fn> \
     --batch-size N --iters 30 --warmup 5 --json /tmp/profile.json
   ```
   For a user's own `train_step`, use `--model script:mymod:train_step`
   (runs `--isolated` by default).
3. **Check validity.** `gpu_event_ms.mean_ms <= wall_ms.mean_ms` must hold.
   If `data_ms / wall_ms > 0.2`, data loading is the bottleneck.
4. **Report.** Quote mean ± std with `n`, samples/sec, memory, and warnings —
   copied from the JSON, with the JSON path cited.

## Reading the numbers

- `wall_ms`: end-to-end per-iteration ms (data + CPU + GPU + optimizer).
- `gpu_event_ms`: GPU kernel time only. Much smaller than wall = GPU is
  starving (host overhead or data loading), not fast.
- `data_ms`: dataloader time between iterations.
- `samples_per_sec`: batch_size / mean wall seconds.

For the full explanation, run with `--explain`.
