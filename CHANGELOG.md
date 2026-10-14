# Changelog

## v0.3

- Bottleneck classification (`analyze.py`): data-loading / host-overhead /
  memory-pressure / variance rules over measured fields only; no MFU/SOL
  (no Apple peak API); Summary/Metrics/Findings report format.
- `mpspulse analyze` CLI with opt-in `--deep` (`torch.mps.profiler` +
  `metalperftrace collect/overview`; missing tools warn, base report
  always emitted).
- Skill `mps-analyze` + OMP command `/analyze-perf`.

## v0.2

- ETA engine (`eta.py`): [p10, p90] intervals from step samples, never a
  single number; `static`/`benchmark`/`live` sources; grad-accum, startup,
  val/ckpt overheads; `~/.mpspulse/history.jsonl` calibration with drift
  warnings.
- `loop.profile_loop(throughput=True)`: ETA calibration via repeated
  uninterrupted runs (per-iter sync timing overstates real async training
  by ~25% — measured on M2 Pro).
- `mpspulse estimate` CLI: static/benchmark/live modes, live reads step
  times from stdin.
- Skill `mps-estimate` + OMP command `/train-estimate`.
- Accuracy proof: `scripts/eta_accuracy.py` 3/3 inside interval, mean
  |err| ~2% (see `examples/mock_experiments/ETA_RESULTS.md`).

## v0.1

- MPS event timing (`timing.benchmark_fn`): reused event pair, per-iter
  sync+read (direct NVIDIA `benchmark_detailed` port is unreliable on MPS —
  see module docstring).
- Training-loop profiling (`loop.profile_loop`): NVIDIA injection template
  port — warmup exclusion, per-iter wall ms, data-load ms, samples/sec,
  optional sub-phase breakdown and GPU event samples.
- `mpspulse profile` CLI: builtin `resnet50`/`vit_b16`/`unet_tiny` models,
  `script:mod:fn` targets, `--isolated` subprocess mode, `--dry-run`
  schema fixture, `--explain`.
- JSON schema (`schema.py`): `gpu_event_ms` vs `wall_ms` always separate.
- Skill `mps-profile` + OMP command `/profile`.
- Tests: pure unit + CLI smoke (torch-free) + MPS-gated timing tests.
