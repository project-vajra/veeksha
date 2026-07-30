"""Unit tests for workload pin check helper."""

from __future__ import annotations

import pytest

from veeksha.named_benchmark.resolve import NamedBenchmarkError, check_workload_pin


@pytest.mark.unit
def test_pin_match_is_silent() -> None:
    check_workload_pin(
        actual_digest="blake2b:abc",
        named_meta={"name": "x", "pins": {"workload_fingerprint": "blake2b:abc"}},
        allow_workload_drift=False,
        stage="preflight",
    )


@pytest.mark.unit
def test_pin_mismatch_raises() -> None:
    with pytest.raises(NamedBenchmarkError, match="mismatch"):
        check_workload_pin(
            actual_digest="blake2b:other",
            named_meta={"name": "x", "pins": {"workload_fingerprint": "blake2b:abc"}},
            allow_workload_drift=False,
            stage="preflight",
        )


@pytest.mark.unit
def test_pin_mismatch_warns_when_drift_allowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    check_workload_pin(
        actual_digest="blake2b:other",
        named_meta={"name": "x", "pins": {"workload_fingerprint": "blake2b:abc"}},
        allow_workload_drift=True,
        stage="finalize",
    )
    assert any("mismatch" in r.message for r in caplog.records)


@pytest.mark.unit
def test_missing_pin_is_noop() -> None:
    check_workload_pin(
        actual_digest="blake2b:abc",
        named_meta={"name": "x", "pins": {}},
        allow_workload_drift=False,
    )
