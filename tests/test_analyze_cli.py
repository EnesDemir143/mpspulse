"""CLI tests for analyze (torch-free paths) + deep fallback."""

import json
import subprocess
import sys

from mpspulse.schema import dry_run_fixture

CLI = [sys.executable, "-m", "mpspulse.cli"]


def test_analyze_clean_bill_dry_run(tmp_path):
    rep = dry_run_fixture()
    rep["wall_ms"] = {
        "mean_ms": 10.0,
        "median_ms": 10.0,
        "std_ms": 0.5,
        "min_ms": 9.0,
        "max_ms": 11.0,
        "n": 30,
    }
    rep["data_ms"] = {
        "mean_ms": 0.1,
        "median_ms": 0.1,
        "std_ms": 0.01,
        "min_ms": 0.05,
        "max_ms": 0.2,
        "n": 30,
    }
    rep["gpu_event_ms"] = {
        "mean_ms": 9.5,
        "median_ms": 9.5,
        "std_ms": 0.4,
        "min_ms": 8.8,
        "max_ms": 10.2,
        "n": 30,
    }
    rep["memory"] = {"current_allocated": 100, "driver_allocated": 200, "recommended_max": 1000}
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(rep))
    out = tmp_path / "out.json"
    proc = subprocess.run(
        [*CLI, "analyze", "--input", str(inp), "--json", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["summary"].startswith("Clean bill")
    assert result["findings"] == []


def test_analyze_data_bottleneck(tmp_path):
    rep = dry_run_fixture()
    rep["wall_ms"] = {
        "mean_ms": 10.0,
        "median_ms": 10.0,
        "std_ms": 0.5,
        "min_ms": 9.0,
        "max_ms": 11.0,
        "n": 30,
    }
    rep["data_ms"] = {
        "mean_ms": 3.0,
        "median_ms": 3.0,
        "std_ms": 0.1,
        "min_ms": 2.8,
        "max_ms": 3.2,
        "n": 30,
    }
    rep["gpu_event_ms"] = None
    rep["memory"] = {"current_allocated": 100, "driver_allocated": 200, "recommended_max": 1000}
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(rep))
    proc = subprocess.run(
        [*CLI, "analyze", "--input", str(inp)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["findings"][0]["class"] == "data-loading"
    assert "[data-loading]" in proc.stderr  # human report on stderr


def test_deep_without_runnable_model_warns_not_fails(tmp_path):
    rep = dry_run_fixture()
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(rep))
    out = tmp_path / "out.json"
    proc = subprocess.run(
        [*CLI, "analyze", "--input", str(inp), "--deep", "--model", "badmodel", "--json", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["deep"]["warnings"] != []
    assert "summary" in result  # base report still emitted
