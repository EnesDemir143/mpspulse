"""Drive the three mock models through real MPS profiling, assert validity.

Usage: uv run scripts/run_mock_suite.py
Asserts per model: JSON written, wall>0, gpu present and <= wall,
repeat CV across 2 runs < 15%, memory keys present.
"""

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOCKS = ["resnet_like", "vit_like", "unet_like"]


def profile_once(mock: str, iters: int = 50, warmup: int = 10) -> dict:
    out = Path(f"/tmp/mpspulse-mock-{mock}.json")
    env = {"PYTHONPATH": str(ROOT / "examples" / "mock_experiments")}
    import os

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mpspulse.cli",
            "profile",
            "--model",
            f"script:{mock}:train_step",
            "--batch-size",
            "8",
            "--iters",
            str(iters),
            "--warmup",
            str(warmup),
            "--json",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env},
        cwd=ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{mock} profile failed: {proc.stderr.strip()[-500:]}")
    return json.loads(out.read_text())


def main() -> int:
    failures: list[str] = []
    for mock in MOCKS:
        try:
            # First run absorbs cold-start noise; CV judged on runs 2-4.
            profile_once(mock)
            runs = [profile_once(mock) for _ in range(3)]
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        r1 = runs[0]
        checks = [
            ("wall>0", r1["wall_ms"]["mean_ms"] > 0),
            ("gpu present", r1["gpu_event_ms"] is not None),
            ("gpu<=wall", r1["gpu_event_ms"]["mean_ms"] <= r1["wall_ms"]["mean_ms"]),
            (
                "memory keys",
                all(
                    r1["memory"][k] > 0
                    for k in ("current_allocated", "driver_allocated", "recommended_max")
                ),
            ),
        ]
        means = [r["wall_ms"]["mean_ms"] for r in runs]
        cv = statistics.stdev(means) / statistics.mean(means)
        checks.append((f"repeat CV {cv:.3f}<0.15", cv < 0.15))
        bad = [name for name, ok in checks if not ok]
        status = "OK " if not bad else "FAIL"
        print(
            f"{status} {mock}: wall={means[0]:.2f}ms gpu="
            f"{r1['gpu_event_ms']['mean_ms']:.2f}ms cv={cv:.3f} "
            f"sps={r1['samples_per_sec']:.0f}"
        )
        failures.extend(f"{mock}: {name}" for name in bad)
    if failures:
        print("FAILURES:", *failures, sep="\n  ")
        return 1
    print("mock suite: all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
