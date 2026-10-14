---
name: mps-sync-trace
description: >
  Find host-device syncs in MPS training code and capture deep GPU traces.
  Use when the user asks about .item()/.cpu() stalls, CPU-GPU sync,
  non_blocking, sync-free code, Metal traces, .gputrace, or gpudebug. Scans
  read-only (never rewrites code); deep capture is opt-in and never slows
  normal profiling.
  Triggers: "find syncs", ".item() in my loop", "sync-free", "host-device
  sync", "capture a Metal trace", "gputrace my training", "inspect .gputrace".
license: Apache-2.0
metadata:
  author: mpspulse contributors
  port_of: NVIDIA perf-torch-sync-free + Apple using-gpucapture/using-gpudebug
---

# MPS Sync & Trace

Two tools: a static sync scanner (always cheap) and deep GPU capture
(explicit opt-in only).

## Sync scan (read-only)

MPS has no runtime sync-debug mode, so detection is a static AST scan for
`.item()`, `.cpu()`, `.numpy()`, `bool(tensor)`, `float(tensor)`, and
`torch.tensor(..., device=...)` in/near loops:

```sh
uv run mpspulse sync-scan path/to/train.py
```

Classify each hit before suggesting fixes:

- **False dependency** (avoidable): debug prints, deferred logging, missing
  `non_blocking=True` on `.to('mps', ...)`.
- **True dependency** (restructure): control flow on GPU values, dynamic
  shapes, CPU-side metric math — move to GPU (`torch.where()`), delay to
  end of iteration, or coalesce into one transfer.

Fix order: remove → `non_blocking=True` → sync-free API → delay → coalesce
→ GPU offload. Never rewrite code automatically.

## Deep trace (opt-in)

```sh
uv run mpspulse gputrace --output trace.gputrace --count 3 --browse -- \
  python train.py
```

- Needs macOS 27+ (`gpucapture`/`gpudebug`); missing tools fail with exit 2
  and a clear message — normal `profile`/`estimate` never set capture env
  vars and stay uncontaminated.
- The workload must run long enough for a capture boundary to appear
  (seconds, not ms).
- `--browse` attaches a non-interactive `gpudebug` summary. Trace metrics
  are graphics-oriented; applied to ML compute they are `experimental`.
