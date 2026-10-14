---
name: mps-analyze
description: >
  Analyze an MPS profile report for bottlenecks. Use when the user asks to
  analyze performance, diagnose a slowdown, check for bottlenecks, or explain
  profiling numbers. Runs the mpspulse CLI on measured JSON — every finding
  traces to evidence, never invented. No MFU/SOL (no Apple peak API).
  Triggers: "analyze this profile", "why is training slow", "diagnose
  bottleneck", "performance analysis", "what is limiting throughput".
license: Apache-2.0
compatibility: Requires the mpspulse repo (uv + Python 3.11+); --deep capture needs macOS on Apple Silicon with torch installed
metadata:
  author: mpspulse contributors
  port_of: NVIDIA perf-analysis
---

# MPS Analyze

Classify bottlenecks from a measured profile report.

## Principles

- **Classify before recommending.** Name the bottleneck class first; the
  wrong classification wastes effort.
- **Evidence for everything.** Each finding cites metric, value, threshold —
  copied from the input JSON.
- **No peak ratios.** MFU / %SOL are omitted: Apple publishes no peak-FLOP
  API to compute them against. Never invent peak numbers.

## Workflow

1. **Analyze.** From the mpspulse repo:
   ```sh
   uv run mpspulse analyze --input /tmp/profile.json
   ```
   Add `--deep` for an opt-in `torch.mps.profiler` + `metalperftrace`
   sample (warnings, never failure, when tools are missing).
2. **Report.** Use the Summary / Metrics / Findings structure from the
   JSON. Quote evidence values; keep recommendations prioritized.

## Bottleneck classes

| Class | Rule (measured only) |
|---|---|
| data-loading | data_ms/wall_ms > 0.2 |
| host-overhead | gpu_event_ms/wall_ms < 0.5 with low data ratio |
| memory-pressure | driver_allocated/recommended_max > 0.85 |
| variance | wall CV > 0.25 (re-run, check shapes/thermals) |
