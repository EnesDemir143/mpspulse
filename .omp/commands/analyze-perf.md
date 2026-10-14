---
description: Analyze an MPS profile report and classify bottlenecks
argument-hint: "<profile.json> [--deep] [--deep-trace -- <cmd>]"

Analyze a measured profile report with the mpspulse CLI.

Request:

$ARGUMENTS

Parse `$ARGUMENTS` as `<profile.json> [--deep]` with an optional
`--deep-trace -- <cmd...>` suffix for full GPU capture.

Workflow:

1. Run in the mpspulse repo:
   `uv run mpspulse analyze --input <profile.json> [--deep]`
2. Present Summary, Metrics, and Findings with evidence (metric, value,
   threshold) from the JSON. Keep recommendations in priority order.
   Numbers from the JSON only — never invent figures.
3. Only when `--deep-trace -- <cmd...>` is present, also run:
   `uv run mpspulse gputrace --browse -- <cmd...>`
   and attach the gpudebug summary labeled experimental. Deep tracing is
   opt-in; never run it unasked.
