---
description: Profile an MPS training step and summarize the measured JSON
argument-hint: "<model> [batch-size] [iters]"
---

Profile a PyTorch MPS training step with the mpspulse CLI and summarize the
measured results.

Request:

$ARGUMENTS

Parse `$ARGUMENTS` as `<model> [batch-size] [iters]` where model is
`resnet50`, `vit_b16`, `unet_tiny`, or `script:module:function`.
Defaults: batch-size 8, iters 30, warmup 5.

Workflow:

1. Run in the mpspulse repo:
   `uv run mpspulse profile --model <model> --batch-size <b> --iters <n> --warmup 5 --json /tmp/mps-<model>.json`
2. Read the JSON. Verify `gpu_event_ms.mean_ms <= wall_ms.mean_ms`; note the
   `data_ms/wall_ms` ratio and any `warnings`.
3. Summarize: mean ± std wall ms (n), gpu event ms, samples/sec, data ratio,
   memory, warnings. Cite the JSON path. Quote numbers from the JSON only —
   never invent figures.
