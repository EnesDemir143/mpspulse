"""CLI smoke tests via --dry-run (no torch needed)."""

import json
import subprocess
import sys

from mpspulse.schema import validate_report

CLI = [sys.executable, "-m", "mpspulse.cli"]


def run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([*CLI, *argv], capture_output=True, text=True, check=False)


def test_help_exit_zero():
    proc = run("profile", "--help")
    assert proc.returncode == 0
    assert "--model" in proc.stdout


def test_unknown_model_exit_two():
    proc = run("profile", "--model", "nope", "--dry-run")
    assert proc.returncode == 0  # dry-run accepts any model name for schema tests
    proc = run("profile", "--model", "nope", "--iters", "1")
    assert proc.returncode == 2
    assert "unknown model" in proc.stderr


def test_dry_run_json_valid_schema(tmp_path):
    out = tmp_path / "r.json"
    proc = run(
        "profile", "--model", "resnet50", "--batch-size", "4", "--dry-run", "--json", str(out)
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out.read_text())
    assert validate_report(report) == []
    assert report["model"] == "resnet50"
    assert report["batch_size"] == 4


def test_explain_flag():
    proc = run("profile", "--model", "resnet50", "--explain")
    assert proc.returncode == 0
    assert "gpu_event_ms" in proc.stdout
    assert "wall_ms" in proc.stdout
