# CUDA → MPS Port Notes

Per-skill mapping decisions from NVIDIA CUDA patterns to Apple Silicon MPS.
Upstream originals: `references/upstream/` (pins in `SOURCES.md`).

## Global mappings

| CUDA / NVIDIA | MPS port | Notes |
|---|---|---|
| `torch.cuda.Event(enable_timing=True)` | `torch.mps.event.Event(enable_timing=True)` | Same record/elapsed API; verify `elapsed_time` nonzero at runtime |
| `torch.cuda.synchronize()` | `torch.mps.synchronize()` | Single sync at end of event benchmark; per-iter sync only in manual loop timing |
| `time.perf_counter()` wall-clock | unchanged | End-to-end: data + CPU + GPU + optimizer |
| nsys / ncu | `torch.mps.profiler` + `metalperftrace` + optional `gpucapture` | Escalation path only; lightweight measurement first |
| NVTX annotate | `torch.mps.profiler.profile` signposts; `std::os_signpost` note | Labels for timelines, never for measurement |
| `torch.cuda.set_sync_debug_mode` | manual static scan (`mpspulse sync-scan`) | verified 2026-10-14 on torch 2.14.1: no `set_sync_debug_mode` under `torch.mps` |
| `torch.cuda.get_device_properties()` / MFU / SOL | omitted | No Apple peak-FLOP/bandwidth API; do NOT invent peak numbers |
| `.to('cuda', non_blocking=True)` | `.to('mps', non_blocking=True)` | Same semantics |
| CUDA Graphs capture regions | omitted | No MPS equivalent; "exclude unavoidable syncs from capture" step dropped |

## perf-workload-profiling → mps-profile (v0.1)

- Keep: warmup → steady-state, hardware event timing, per-iteration stats
  (mean/median/std/min/max), data/iter > 0.2 rule.
- Deviation (verified 2026-10-14, torch 2.14.1): the direct `benchmark_detailed`
  port (2n pre-allocated pairs, one final sync, bulk read) fails intermittently
  on MPS (~15%: "End event was not recorded after start event"). mpspulse uses
  a single reused event pair with per-iteration sync + immediate read
  (0 failures in testing). The sync sits after `end.record()`, so measured
  GPU time excludes sync overhead.
- Loop template: `torch.cuda.synchronize` → `torch.mps.synchronize`, rest identical.
- `benchmark_fn`: MPS device only, never silent CPU fallback (corrupts GPU/wall split).
- Sub-phase breakdown: optional sync+perf_counter per phase (one-off diagnosis).
- Output: JSON with separate `gpu_event_ms` / `wall_ms` keys (schema.py).

## perf-analysis → mps-analyze (v0.3)

- Keep: measure-never-invent, classify-before-recommending, Summary/Metrics/
  Findings/Recommendations report format.
- Drop: specialist delegation (single-machine local analysis), MFU/SOL metrics
  (no peak API), communication-bound class (single GPU), SLURM/remote.
- Bottleneck rules operate ONLY on measured fields:
  `data_ms/wall_ms > 0.2` → data-loading; `gpu_event_ms/wall_ms < 0.5` →
  host-overhead; driver memory > 85% recommended → memory-pressure;
  wall CV > 25% → variance suspect.
- `--deep` escalation: `torch.mps.profiler` + `metalperftrace` if present;
  missing tools → warning, base report still emitted.

## perf-torch-sync-free → mps-sync-trace (v0.4)

- Detection: AST static scan for `.item()`, `.cpu()`, `.numpy()`,
  `bool(tensor)`, `float(tensor)`, cross-device tensor creation in/near loops.
- No runtime sync-debug mode on MPS (verified absent on torch 2.14.1, 2026-10-14).
- Keep false-vs-true dependency classification and elimination ordering
  (remove → non_blocking → sync-free API → delay → coalesce → GPU offload).
- Read-only: never rewrites code.
- `gpucapture`/`gpudebug` workflow ported from Apple skills; opt-in only,
  never contaminates normal profile/estimate paths. Graphics-oriented metrics
  applied to ML compute labeled `experimental`.

## Tool flags — verify from local man pages at implementation

- `metalperftrace`: flags **unverified — confirm first** via `man metalperftrace`.
- `gpucapture start --until-exit`: exact flags **unverified — confirm first**
  via `man gpucapture`.
- `gpudebug -q` scripted browse: per `man gpudebug`.
