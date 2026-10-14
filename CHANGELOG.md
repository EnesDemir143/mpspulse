# Changelog

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
