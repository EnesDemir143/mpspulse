"""CLI tests for sync-scan and gputrace (torch-free)."""

import json
import os
import subprocess
import sys
from pathlib import Path

CLI = [sys.executable, "-m", "mpspulse.cli"]
FIXTURES = Path(__file__).parent / "fixtures"


def test_sync_scan_lists_findings():
    proc = subprocess.run(
        [*CLI, "sync-scan", str(FIXTURES / "syncy.py")], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    findings = json.loads(proc.stdout)
    assert len(findings) >= 3
    assert findings[0]["line"] == 12
    assert "syncy.py:12" in proc.stderr


def test_sync_scan_clean_file():
    proc = subprocess.run(
        [*CLI, "sync-scan", str(FIXTURES / "clean.py")], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == []


def test_gputrace_missing_binary_exit_two():
    env = {**os.environ, "PATH": "/tmp/notools:/bin"}
    Path("/tmp/notools").mkdir(exist_ok=True)
    proc = subprocess.run(
        [*CLI, "gputrace", "--", "echo", "hi"], capture_output=True, text=True, check=False, env=env
    )
    assert proc.returncode == 2
    assert "gpucapture not found" in proc.stderr


def test_gputrace_needs_command():
    proc = subprocess.run([*CLI, "gputrace"], capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert "needs a command" in proc.stderr
