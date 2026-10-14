"""Pure unit tests for bottleneck classification (no torch)."""

import copy

import pytest

from mpspulse.analyze import analyze, classify
from mpspulse.schema import dry_run_fixture


def _report(**overrides) -> dict:
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
    for key, value in overrides.items():
        if isinstance(value, dict) and key in rep:
            rep[key] = {**rep[key], **value}
        else:
            rep[key] = value
    return rep


def test_clean_bill():
    assert classify(_report()) == []
    assert "Clean bill" in analyze(_report()).summary


def test_data_loading_bottleneck():
    rep = _report(data_ms={"mean_ms": 3.0})
    (finding,) = classify(rep)
    assert finding.class_ == "data-loading"
    assert finding.to_dict()["evidence"]["value"] == pytest.approx(0.3)
    assert finding.priority == 1


def test_host_overhead_bottleneck():
    rep = _report(gpu_event_ms={"mean_ms": 3.0})
    (finding,) = classify(rep)
    assert finding.class_ == "host-overhead"
    assert "sync-scan" in finding.recommendation


def test_host_overhead_suppressed_when_data_dominates():
    rep = _report(gpu_event_ms={"mean_ms": 3.0}, data_ms={"mean_ms": 3.0})
    classes = [f.class_ for f in classify(rep)]
    assert classes == ["data-loading"]


def test_memory_pressure():
    rep = _report(memory={"driver_allocated": 900})
    (finding,) = classify(rep)
    assert finding.class_ == "memory-pressure"
    assert finding.to_dict()["evidence"]["value"] == pytest.approx(0.9)


def test_variance_suspect():
    rep = _report(wall_ms={"mean_ms": 10.0, "std_ms": 3.0, "n": 30})
    (finding,) = classify(rep)
    assert finding.class_ == "variance"
    assert "re-run" in finding.recommendation.lower()


def test_priority_order_and_evidence_copied():
    rep = _report(data_ms={"mean_ms": 3.0}, memory={"driver_allocated": 900})
    findings = classify(rep)
    assert [f.class_ for f in findings] == ["data-loading", "memory-pressure"]
    assert findings[0].to_dict()["evidence"]["value"] == pytest.approx(0.3)


def test_analyze_report_shape():
    result = analyze(_report(data_ms={"mean_ms": 3.0})).to_dict()
    assert result["summary"].startswith("Primary bottleneck: data-loading")
    assert result["metrics"]["data_ratio"] == pytest.approx(0.3)
    assert result["findings"][0]["class"] == "data-loading"
    assert copy.deepcopy(result) == result  # JSON-serializable shape
