# Mock Experiment Results (v0.1 level)

Device: Apple M2 Pro, macOS 27.0.1, torch 2.14.1 (MPS)
Date: 2026-10-14
Suite: `uv run scripts/run_mock_suite.py` — each mock profiled 4× via
`mpspulse profile --model script:<mock>:train_step --iters 50 --warmup 10`;
first run discarded as cold start, CV computed over runs 2–4.

| Mock | Wall mean ms | GPU event mean ms | samples/sec | Repeat CV |
|---|---|---|---|---|
| resnet_like | 8.33 | 8.14 | 961 | 0.045 |
| vit_like | 6.99 | 6.80 | 1145 | 0.039 |
| unet_like | 3.83 | 3.70 | 2086 | 0.020 |

All assertions green: JSON written, wall > 0, gpu present and ≤ wall,
repeat CV < 0.15, memory keys nonzero.

Note: with the original `--iters 20 --warmup 5` and 2 runs, small mocks
(3–7 ms) intermittently exceeded CV 0.15 under background machine load.
The suite now uses 50 iters and drops the first run; the criterion
(CV < 0.15) is unchanged.
