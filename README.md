# mpspulse — Agent Skills for Apple Silicon PyTorch Profiling

NVIDIA TensorRT-LLM performance skills, ported to macOS Apple Silicon PyTorch
MPS: real measurements via a Python CLI, guidance via Agent Skills. Works with
any agent that supports the [Agent Skills](https://agentskills.io/) format
(Claude Code, Cursor, Codex, OpenCode, and more).

[![skills.sh](https://skills.sh/b/EnesDemir143/mpspulse)](https://skills.sh/EnesDemir143/mpspulse)

## Install

```sh
# 1. Install the skills into your agent
npx skills add EnesDemir143/mpspulse

# 2. Install the measurement CLI (skills call this; they never invent numbers)
pip install "mpspulse[mps]"
```

The base `pip install mpspulse` is torch-free (~MBs: `analyze`, `sync-scan`,
`estimate --mode static/live` all work). The `[mps]` extra adds PyTorch
(~2 GB) for on-device measurement.

## Available Skills

### mps-profile

Profile a representative PyTorch MPS training step with warmup, steady-state
timing, and GPU-vs-wall separation.

**Use when:** "profile this model", "benchmark my training step", "samples
per second", "how slow is my loop", "data load time", "MPS timing".

### mps-estimate

Estimate total training time from measured step times. Reports an interval
with source and warnings, never a single invented number.

**Use when:** "how long will training take", "estimate training time", "ETA
for my run", "predict runtime", "time to train".

### mps-analyze

Classify bottlenecks (data-loading / host-overhead / memory-pressure /
variance) from a measured profile report. Every finding traces to evidence.

**Use when:** "analyze this profile", "why is training slow", "diagnose
bottleneck", "performance analysis", "what is limiting throughput".

### mps-sync-trace

Find host-device syncs (`.item()`, `.cpu()`, …) in training code via a
read-only static scan, and optionally capture deep `.gputrace` GPU traces.

**Use when:** "find syncs", ".item() in my loop", "sync-free",
"host-device sync", "capture a Metal trace", "gputrace my training".

## How it works

```
You ask → skill triggers → skill runs `mpspulse` CLI → real JSON → report
```

- **Skill = guidance; CLI = measurement.** The agent never invents numbers.
- **GPU event time and wall-clock time are always separate.** `gpu_event_ms`
  (hardware GPU time) vs `wall_ms` (end-to-end) vs `data_ms` (dataloader).
- **No silent CPU fallback.** Without MPS, measurement exits 2 with a clear
  message instead of corrupting GPU/wall separation.
- **`script:` targets profile `--isolated`** in a subprocess by default, so
  your model/optimizer state is untouched.

## CLI reference

```sh
mpspulse profile --model resnet50 --batch-size 8 --iters 30 --json /tmp/r.json
mpspulse estimate --steps 10000 --mode benchmark --model resnet50
mpspulse analyze --input /tmp/r.json [--deep]
mpspulse sync-scan path/to/train.py
mpspulse gputrace --output trace.gputrace --count 3 --browse -- python train.py
```

Builtin models (`resnet50`, `vit_b16`, `unet_tiny`) are hand-rolled
torchvision-free definitions; profile your own code with
`--model script:mymodule:train_step`.

## Repo layout

```text
skills/     # SKILL.md files (Agent Skills format, npx-skills installable)
src/        # mpspulse CLI: timing, loop profiling, ETA, analysis
tests/      # unit + MPS-gated integration tests
references/ # upstream pins (SOURCES.md), CUDA→MPS port notes (PORT_NOTES.md)
examples/   # mock experiment harness + measured M2 Pro results
scripts/    # validation scripts (mock suite, ETA accuracy)
```

## Development

```sh
uv sync --group dev       # lint + test env (no torch)
uv sync --extra mps       # + PyTorch for on-device measurement
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

## License

Apache-2.0 (see LICENSE, NOTICE). Upstream pins: `references/SOURCES.md`.
