# Upstream Sources

Vendored reference material. All files under `references/upstream/` are
unmodified copies; MPS ports live in `skills/` and `src/`.

Fetch date: 2026-10-14.

## NVIDIA TensorRT-LLM performance skills

Repo: https://github.com/NVIDIA/TensorRT-LLM (dir `.claude/skills/`)
Commit: `91404de7bcbcf14900684547001de48b8d50e5f7` (`main` HEAD, 2026-10-14)
License: Apache-2.0, © NVIDIA Corporation

| File | Upstream path |
|---|---|
| `upstream/nvidia/perf-workload-profiling/SKILL.md` | `perf-workload-profiling/SKILL.md` |
| `upstream/nvidia/perf-workload-profiling/references/benchmarking-patterns.md` | `perf-workload-profiling/references/benchmarking-patterns.md` |
| `upstream/nvidia/perf-workload-profiling/references/nvtx-api.md` | `perf-workload-profiling/references/nvtx-api.md` |
| `upstream/nvidia/perf-workload-profiling/references/pytorch-profiler-api.md` | `perf-workload-profiling/references/pytorch-profiler-api.md` |
| `upstream/nvidia/perf-analysis/SKILL.md` | `perf-analysis/SKILL.md` |
| `upstream/nvidia/perf-torch-sync-free/SKILL.md` | `perf-torch-sync-free/SKILL.md` |
| `upstream/nvidia/perf-torch-sync-free/references/sync-patterns.md` | `perf-torch-sync-free/references/sync-patterns.md` |

## Apple game-porting-toolkit skills

Repo: https://github.com/apple/game-porting-toolkit (dir `game-porting-skills/skills/`)
Commit: `4f42344c90be22112f46becc3fb9687f423f2f88` (`main` HEAD, 2026-10-14)
License: see upstream repo (© 2026 Apple Inc.)

| File | Upstream path |
|---|---|
| `upstream/apple/using-gpucapture/SKILL.md` | `using-gpucapture/SKILL.md` |
| `upstream/apple/using-gpudebug/SKILL.md` | `using-gpudebug/SKILL.md` |

## PyTorch

Pinned via `uv.lock`: `torch==2.14.1` (PyPI latest, 2026-10-14).
Optional extra `mps` so base install stays torch-free.
