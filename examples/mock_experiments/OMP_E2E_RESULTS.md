# OMP End-to-End Results

Device: Apple M2 Pro, macOS 27.0.1. Each command run via `omp -p` with `--cwd`=repo and `.omp/` staged from canonical `skills/`+`commands/`. Every ms/sec figure in the output must appear (within 5% rounding) in the referenced JSON artifact.

## /profile: green

```
$ omp -p '/profile unet_tiny 2 10'
(exit 0)
### `unet_tiny` — MPS profile

Batch size **2**, **10** measured iterations, **5** warmup iterations.

| Metric | Measured result |
|---|---:|
| Wall time, mean ± std | **26.0348 ± 2.7084 ms** (n = 10) |
| GPU event time, mean ± std | **25.7721 ± 2.6669 ms** (n = 10) |
| Throughput | **76.8203 samples/sec** |
| Data time, mean | **0.0113168 ms** |
| Data / wall ratio | **0.043468%**, calculated from JSON means |
| Current allocated memory | **21,595,648 bytes** |
| Driver allocated memory | **169,525,248 bytes** |
| Recommended maximum memory | **12,713,115,648 bytes** |

**Timing check passed:** GPU mean **25.7721 ms ≤ 26.0348 ms** wall mean. Data loading is below the skill’s bottleneck threshold.

**Warnings:** JSON contains none (`[]`). The CLI separately emitted a PyTorch warning that NumPy initialization failed because `numpy` is not installed.

Source: [`/tmp/mps-unet_tiny.json`](/tmp/mps-unet_tiny.json).
Working...


```

## /train-estimate: green

```
$ omp -p '/train-estimate 500 --mode benchmark --model unet_tiny --batch-size 2'
(exit 0)
**Estimated total training time:** p50 **6.756970483305244 s**, with **[p10, p90] = [6.675687981575416 s, 6.857845000146578 s]**.

- Configuration: `500` steps, `unet_tiny`, batch size `2`
- Source: `benchmark`
- n_samples: `10`
- JSON warnings: `[]`

Runtime warning, verbatim:
```text
/Users/enesdemir/Documents/mpspulse/.venv/lib/python3.14/site-packages/torch/_subclasses/functional_tensor.py:368: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
```
Working...


```

## /analyze-perf: green

```
$ omp -p '/analyze-perf /tmp/mps-unet_tiny.json'
(exit 0)
## Summary

**No bottleneck detected in the measured fields.** `mpspulse analyze` completed successfully for `unet_tiny` on MPS.

The profile contains **10 measured iterations**, **5 warmup iterations**, and **batch size 2**. This result does not establish whether GPU execution is compute-bound or memory-bandwidth-bound.

## Metrics

Values below are copied from the profile or the analyzer’s JSON output.

| Metric | Value |
|---|---:|
| Mean wall time | 26.034770801197737 ms |
| Wall-time standard deviation | 2.7083619925305302 ms |
| Mean GPU-event time | 25.7720958 ms |
| Mean data-loading time | 0.011316794552840292 ms |
| Throughput | 76.82034212139057 samples/s |
| Data / wall ratio | 0.0004346800146333403 |
| GPU-event / wall ratio | 0.9899106082706265 |
| Current allocated memory | 21595648 bytes |
| Driver-allocated memory | 169525248 bytes |
| Recommended maximum memory | 12713115648 bytes |

## Findings

- **None:** the analyzer returned `"findings": []`; the profile contains `"warnings": []`.
- **Threshold evidence:** neither JSON includes threshold values. No finding-specific metric/value/threshold evidence was emitted.
- **Recommendations:** no corrective optimization is indicated by this analysis. GPU-event time closely tracks wall time; data-loading time is small relative to wall time.

Deep analysis and experimental GPU capture were **not run**; neither was requested.
Working...


```
