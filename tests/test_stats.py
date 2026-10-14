"""Pure unit tests: stats + schema (no torch)."""

import pytest

from mpspulse.schema import dry_run_fixture, validate_report
from mpspulse.stats import cv, summarize


def test_summarize_stats():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s == {
        "mean_ms": 2.5,
        "median_ms": 2.5,
        "std_ms": pytest.approx(1.291, abs=1e-3),
        "min_ms": 1.0,
        "max_ms": 4.0,
        "n": 4,
    }


def test_summarize_single_sample_std_zero():
    assert summarize([5.0])["std_ms"] == 0.0


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


def test_cv_degenerate():
    assert cv([]) == 0.0
    assert cv([3.0]) == 0.0
    assert cv([2.0, 2.0, 2.0]) == 0.0


def test_dry_run_fixture_valid_schema():
    assert validate_report(dry_run_fixture()) == []


def test_validate_report_catches_missing_keys():
    errors = validate_report({"tool": "mpspulse"})
    assert any("wall_ms" in e for e in errors)
    assert any("memory" in e for e in errors)


def test_validate_report_rejects_wrong_tool():
    bad = dry_run_fixture()
    bad["tool"] = "other"
    assert any("tool" in e for e in validate_report(bad))
