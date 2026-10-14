"""Pure unit tests for the ETA engine (no torch)."""

import pytest

from mpspulse.eta import (
    EtaTracker,
    drift_warning,
    estimate,
    lookup_history,
    record_history,
)


def test_interval_math():
    rep = estimate(100, [10.0] * 20)
    assert rep.p50_s == pytest.approx(1.0)
    assert rep.p10_s == pytest.approx(1.0)
    assert rep.p90_s == pytest.approx(1.0)
    assert rep.n_samples == 20


def test_interval_spread_and_order():
    rep = estimate(100, [float(x) for x in range(10, 110)])
    assert rep.p10_s <= rep.p50_s <= rep.p90_s
    assert rep.p10_s < rep.p90_s


def test_overheads_and_grad_accum():
    rep = estimate(100, [10.0] * 10, startup_s=5.0, val_overhead_per_epoch_s=2.0, epochs=3)
    assert rep.p50_s == pytest.approx(1.0 + 5.0 + 6.0)
    rep = estimate(100, [10.0] * 10, grad_accum=4)
    assert rep.p50_s == pytest.approx(4.0)


def test_static_warns_low_confidence():
    rep = estimate(10, [5.0] * 10, source="static")
    assert any("low-confidence" in w for w in rep.warnings)
    rep = estimate(10, [5.0] * 10, source="benchmark")
    assert not any("low-confidence" in w for w in rep.warnings)


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        estimate(10, [])
    with pytest.raises(ValueError):
        estimate(0, [1.0])
    with pytest.raises(ValueError):
        estimate(10, [1.0], grad_accum=0)


def test_tracker_incremental():
    tracker = EtaTracker(100)
    tracker.update(10.0)
    rep = tracker.update(10.0)
    assert rep.source == "live"
    assert rep.n_samples == 2
    assert rep.p50_s == pytest.approx(1.0)


def test_history_drift_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("MPSPULSE_HOME", str(tmp_path))
    record_history("m", "cfg", 10.0, 20)
    record_history("m", "cfg", 11.0, 20)
    assert lookup_history("m", "cfg") != []
    assert drift_warning("m", "cfg", 10.5) is None
    warn = drift_warning("m", "cfg", 20.0)
    assert warn is not None and "drift" in warn
    assert drift_warning("m", "other", 99.0) is None
