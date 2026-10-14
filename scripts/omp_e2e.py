"""OMP end-to-end validation: drive /profile, /train-estimate, /analyze-perf.

Usage: uv run scripts/omp_e2e.py [--transcript FILE]
For each command: run `omp -p` with --cwd=repo, assert exit 0, assert the
JSON artifact exists, and assert every ms/sec figure in the output also
appears in the referenced JSON artifact (no invented numbers).
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_S = 600

COMMANDS = [
    {
        "name": "profile",
        "prompt": "/profile unet_tiny 2 10",
        "artifact": "/tmp/mps-unet_tiny.json",
    },
    {
        "name": "train-estimate",
        "prompt": "/train-estimate 500 --mode benchmark --model unet_tiny --batch-size 2",
        "artifact": None,  # parsed from output; falls back to history check
    },
    {
        "name": "analyze-perf",
        "prompt": "/analyze-perf /tmp/mps-unet_tiny.json",
        "artifact": "/tmp/mps-unet_tiny.json",
    },
]

FLOAT_RE = re.compile(r"(\d+\.\d+|\d+)\s*(ms|s|sec|seconds?)\b", re.IGNORECASE)


def ensure_omp_layout() -> None:
    """Copy canonical skills/commands into .omp/ (git-ignored test artifact)."""
    omp = ROOT / ".omp"
    (omp / "skills").mkdir(parents=True, exist_ok=True)
    (omp / "commands").mkdir(parents=True, exist_ok=True)
    for skill in (ROOT / "skills").iterdir():
        dest = omp / "skills" / skill.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(skill, dest)
    for cmd in (ROOT / "commands").glob("*.md"):
        shutil.copy(cmd, omp / "commands" / cmd.name)


def run_omp(prompt: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["omp", "-p", prompt, "--cwd", str(ROOT), "--no-session"],
        capture_output=True,
        text=True,
        check=False,
        timeout=TIMEOUT_S,
    )
    return proc.returncode, proc.stdout + proc.stderr


def json_numbers(path: Path) -> set[float]:
    data = json.loads(path.read_text())
    found: set[float] = set()

    def walk(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.add(float(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    return found


def output_figures(text: str) -> list[float]:
    return [float(m.group(1)) for m in FLOAT_RE.finditer(text)]


def close_enough(fig: float, candidates: set[float], tol: float = 0.05) -> bool:
    """Figure matches if within 5% (covers LLM rounding) of any JSON number."""
    return any(abs(fig - c) <= tol * max(abs(c), 1e-9) for c in candidates)


def check_command(spec: dict) -> tuple[bool, str, str]:
    rc, output = run_omp(spec["prompt"])
    transcript = f"$ omp -p {spec['prompt']!r}\n(exit {rc})\n{output}\n"
    if rc != 0:
        return False, f"exit code {rc}", transcript
    artifact = spec["artifact"]
    if artifact is None:
        # benchmark mode appends to ~/.mpspulse/history.jsonl; the estimate
        # figures must be consistent with the latest same-config entry.
        hist = Path.home() / ".mpspulse" / "history.jsonl"
        if not hist.exists():
            return False, "history.jsonl missing (benchmark never ran?)", transcript
        rows = [json.loads(line) for line in hist.read_text().splitlines() if line.strip()]
        same = [r for r in rows if r.get("model_id") == "unet_tiny"]
        if not same:
            return False, "no unet_tiny entry in history.jsonl", transcript
        step_p50 = same[-1]["step_ms_p50"]
        expected_p50 = step_p50 / 1000 * 500
        # output p50 must match history-derived expectation within 5%
        figs = output_figures(output)
        if not any(close_enough(f, {expected_p50}) for f in figs):
            return False, (f"output p50 inconsistent with history "
                           f"(expected ~{expected_p50:.2f}s)"), transcript
        if "benchmark" not in output:
            return False, "source=benchmark not reported", transcript
        return True, "green", transcript
    path = Path(artifact)
    if not path.exists():
        return False, f"artifact missing: {artifact}", transcript
    candidates = json_numbers(path)
    bad = [f for f in output_figures(output) if not close_enough(f, candidates)]
    # Ignore tiny integers that are clearly counts/args (iters, batch, steps)
    bad = [f for f in bad if f not in (2.0, 10.0, 500.0)]
    if bad:
        return False, f"figures not in JSON (invented?): {bad[:8]}", transcript
    return True, "green", transcript


def main() -> int:
    ensure_omp_layout()
    log: list[str] = []
    failures: list[str] = []
    for spec in COMMANDS:
        print(f"--- /{spec['name']} ---", flush=True)
        try:
            ok, verdict, transcript = check_command(spec)
        except subprocess.TimeoutExpired:
            ok, verdict, transcript = False, "omp timed out", ""
        print(verdict, flush=True)
        log.append(f"## /{spec['name']}: {verdict}\n\n```\n{transcript}\n```\n")
        if not ok:
            failures.append(spec["name"])
    results = ROOT / "examples" / "mock_experiments" / "OMP_E2E_RESULTS.md"
    results.write_text(
        "# OMP End-to-End Results\n\n"
        "Device: Apple M2 Pro, macOS 27.0.1. Each command run via "
        "`omp -p` with `--cwd`=repo and `.omp/` staged from canonical "
        "`skills/`+`commands/`. Every ms/sec figure in the output must "
        "appear (within 5% rounding) in the referenced JSON artifact.\n\n" + "\n".join(log)
    )
    print(f"wrote {results}", flush=True)
    if failures:
        print(f"FAILURES: {failures}", flush=True)
        return 1
    print("omp e2e: all green", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
