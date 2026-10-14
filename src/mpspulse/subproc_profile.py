"""Subprocess entry for `--isolated` profiling.

Usage: python -m mpspulse.subproc_profile <model> <batch> <iters> <warmup>
Prints the report JSON to stdout. Runs the measurement in a fresh process
so the caller's model/optimizer/checkpoint state is untouched.
"""

import argparse
import json
import sys

from mpspulse.cli import run_profile


def main() -> int:
    ns = argparse.Namespace(
        model=sys.argv[1],
        batch_size=int(sys.argv[2]),
        iters=int(sys.argv[3]),
        warmup=int(sys.argv[4]),
    )
    try:
        report = run_profile(ns)
    except Exception as exc:  # noqa: BLE001 — subprocess boundary, report via stderr
        print(f"mpspulse: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
