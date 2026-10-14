---
description: Analyze an MPS profile report and classify bottlenecks
argument-hint: "<profile.json> [--deep]"
---

Analyze a measured profile report with the mpspulse CLI.

Request:

$ARGUMENTS

Parse `$ARGUMENTS` as `<profile.json> [--deep]`; pass `--deep` through to
the CLI when present.

Workflow:

1. Run in the mpspulse repo:
   `uv run mpspulse analyze --input <profile.json> [--deep]`
2. Present Summary, Metrics, and Findings with evidence (metric, value,
   threshold) from the JSON. Keep recommendations in priority order.
   Numbers from the JSON only — never invent figures.
