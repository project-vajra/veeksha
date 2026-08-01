"""Unit tests for harness request-lifecycle drift metrics + soft WARN."""

import logging
from types import SimpleNamespace

import pytest

from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.evaluator.performance.base import PerformanceEvaluator


def _response(ready, dispatched, pickup, sent):
    return SimpleNamespace(
        scheduler_ready_at=ready,
        scheduler_dispatched_at=dispatched,
        client_picked_up_at=pickup,
        client_sent_at=sent,
    )


def _evaluator(threshold_ms=5.0):
    return PerformanceEvaluator(
        PerformanceEvaluatorConfig(lifecycle_drift_warn_threshold_ms=threshold_ms)
    )


@pytest.mark.unit
def test_lifecycle_metrics_in_summary():
    ev = _evaluator()
    for _ in range(20):
        # ready->dispatch 1ms, dispatch->pickup 0.5ms, pickup->send 2ms
        ev._accumulate_lifecycle_drift(_response(100.0, 100.001, 100.0015, 100.0035))

    summary = ev.get_aggregated_summary()
    assert "Harness Ready-to-Dispatch (ms) (P99)" in summary
    assert summary["Harness Ready-to-Dispatch (ms) (P99)"] == pytest.approx(
        1.0, rel=0.05
    )
    assert summary["Harness Pickup-to-Send (ms) (P99)"] == pytest.approx(2.0, rel=0.05)
    assert summary["Harness Ready-to-Send (ms) (P99)"] == pytest.approx(3.5, rel=0.05)


@pytest.mark.unit
def test_missing_timestamps_are_skipped():
    ev = _evaluator()
    # client_sent_at missing -> pickup->send and ready->send have no samples
    ev._accumulate_lifecycle_drift(_response(100.0, 100.001, 100.0015, None))
    assert ev._lifecycle_sketches["Harness Ready-to-Dispatch (ms)"].sketch.count == 1
    assert ev._lifecycle_sketches["Harness Pickup-to-Send (ms)"].sketch.count == 0
    assert ev._lifecycle_sketches["Harness Ready-to-Send (ms)"].sketch.count == 0


@pytest.mark.unit
def test_warn_fires_above_threshold(caplog):
    ev = _evaluator(threshold_ms=5.0)
    for _ in range(30):
        # pickup->send = 20ms, well above the 5ms threshold
        ev._accumulate_lifecycle_drift(_response(100.0, 100.001, 100.0015, 100.0215))
    with caplog.at_level(logging.WARNING):
        ev._warn_on_lifecycle_drift()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Pickup-to-Send" in w and "lifecycle drift high" in w for w in warnings)


@pytest.mark.unit
def test_no_warn_within_threshold(caplog):
    ev = _evaluator(threshold_ms=50.0)
    for _ in range(30):
        ev._accumulate_lifecycle_drift(_response(100.0, 100.001, 100.0015, 100.0035))
    with caplog.at_level(logging.WARNING):
        ev._warn_on_lifecycle_drift()
    assert not [r for r in caplog.records if "lifecycle drift high" in r.getMessage()]


@pytest.mark.unit
def test_warn_disabled_when_threshold_nonpositive(caplog):
    ev = _evaluator(threshold_ms=0.0)
    for _ in range(30):
        ev._accumulate_lifecycle_drift(_response(100.0, 100.001, 100.0015, 100.5))
    with caplog.at_level(logging.WARNING):
        ev._warn_on_lifecycle_drift()
    assert not [r for r in caplog.records if "lifecycle drift high" in r.getMessage()]
