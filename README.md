# mpspulse

Apple Silicon PyTorch MPS profiling and training ETA. NVIDIA TensorRT-LLM
performance skills, ported to macOS MPS: real measurements via a Python CLI,
guidance via Agent Skills + OMP slash commands.

## Install

```sh
uv sync                  # CLI + tests, no torch (~MBs)
uv sync --extra mps      # + PyTorch with MPS (~2 GB)
```

## Quickstart

```sh
uv run mpspulse profile --model resnet50 --batch-size 8 --iters 30 \
  --json /tmp/r.json
uv run mpspulse profile --model resnet50 --explain   # GPU-vs-wall guide
```

## GPU event time vs wall-clock time

- `gpu_event_ms`: hardware-measured GPU kernel time (MPS event pairs).
- `wall_ms`: end-to-end per-iteration time (data + CPU + GPU + optimizer).
- `data_ms`: dataloader time between iterations; `data/wall > 0.2` means
  data loading is the bottleneck.
- `samples_per_sec`: batch_size / mean wall seconds.

A `gpu_event_ms` much smaller than `wall_ms` means the GPU is idle most of
the iteration (host overhead or data starvation).

## Layout

```text
skills/     # SKILL.md files (Agent Skills compatible, OMP native discovery)
commands/   # OMP slash commands (/profile, /train-estimate, /analyze-perf)
src/        # timing, loop profiling, ETA, analysis (CLI does real measurement)
tests/      # unit + MPS-gated integration tests
references/ # upstream pins (SOURCES.md), CUDA→MPS port notes (PORT_NOTES.md)
examples/   # mock experiment harness + measured results
scripts/    # validation scripts (mock suite, ETA accuracy, OMP e2e)
```

## Rules

- Skill = guidance; Python CLI = measurement. The LLM never invents numbers.
- GPU event time and wall-clock time are always reported separately.
- Benchmarks never silently fall back to CPU (exit 2 with a clear message).
- `script:` targets profile `--isolated` in a subprocess by default.

## License

Apache-2.0 (see LICENSE, NOTICE). Upstream pins: `references/SOURCES.md`.
