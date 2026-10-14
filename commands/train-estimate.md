---
description: Estimate total MPS training time from measured step times
argument-hint: "<steps> [--mode benchmark]"
---

Estimate total training time with the mpspulse CLI and report an interval.

Request:

$ARGUMENTS

Parse `$ARGUMENTS` as `<steps> [options]` where options may include
`--mode benchmark --model <m> --batch-size <b> --epochs <e> --step-ms <x>`
(pass through unknown flags to the CLI).

Workflow:

1. Run in the mpspulse repo, e.g.:
   `uv run mpspulse estimate --steps <steps> --mode benchmark --model <model>`
   Default mode is `static` (needs `--step-ms`); prefer `benchmark` when a
   model is named.
2. Read the JSON. Report p50 with [p10, p90], source, n_samples, and every
   warning verbatim. Numbers from the JSON only — never invent figures.
