"""Tests for TTS-specific health checks (truncation, zombie sessions)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.evaluator import (
    AudioChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.health import HealthChecker


def _write_metrics(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as file:
        for row in rows:
            # Baseline lifecycle columns consumed by the generic health checks.
            full_row = {
                "session_id": row["request_id"],
                "scheduler_dispatched_at": float(row["request_id"]),
                "client_completed_at": float(row["request_id"]) + 1.0,
                **row,
            }
            file.write(json.dumps(full_row) + "\n")


def _benchmark_config(max_expected_audio_ms: float | None) -> BenchmarkConfig:
    return BenchmarkConfig(
        evaluators=[
            PerformanceEvaluatorConfig(
                target_channels=["audio"],
                slos=[],
                audio_channel=AudioChannelPerformanceConfig(
                    max_expected_audio_ms=max_expected_audio_ms,
                ),
            )
        ],
    )


def _health_checker(
    tmp_path: Path,
    rows: list[dict],
    max_expected_audio_ms: float | None,
) -> HealthChecker:
    metrics_file = tmp_path / "request_level_metrics.jsonl"
    _write_metrics(metrics_file, rows)
    checker = HealthChecker(
        trace_file=str(tmp_path / "missing_trace.jsonl"),
        metrics_file=str(metrics_file),
        benchmark_config=_benchmark_config(max_expected_audio_ms),
    )
    assert checker.load_data()
    return checker


# ---------------------------------------------------------------------------
# Suspected length-cap truncation check
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_truncation_check_fails_when_requests_are_flagged(tmp_path: Path) -> None:
    checker = _health_checker(
        tmp_path,
        rows=[
            {"request_id": 1, "suspected_length_cap_truncation": 1},
            {"request_id": 2, "suspected_length_cap_truncation": 0},
        ],
        max_expected_audio_ms=163840.0,
    )

    result = checker.check_suspected_length_cap_truncation()

    assert result.passed is False
    results = result.summary["sections"][0]["results"]
    assert results["Suspected Truncations"].startswith("1 ")
    assert results["Sample Request IDs"] == "1"


@pytest.mark.unit
def test_truncation_check_passes_when_no_requests_are_flagged(
    tmp_path: Path,
) -> None:
    checker = _health_checker(
        tmp_path,
        rows=[
            {"request_id": 1, "suspected_length_cap_truncation": 0},
            {"request_id": 2, "suspected_length_cap_truncation": 0},
        ],
        max_expected_audio_ms=163840.0,
    )

    result = checker.check_suspected_length_cap_truncation()

    assert result.passed is True


@pytest.mark.unit
def test_truncation_check_skips_when_flag_column_missing(tmp_path: Path) -> None:
    checker = _health_checker(
        tmp_path,
        rows=[{"request_id": 1, "ttfc": 100.0}],
        max_expected_audio_ms=163840.0,
    )

    result = checker.check_suspected_length_cap_truncation()

    assert result.passed is True
    assert result.summary["sections"][0]["results"]["Status"] == "Skipped"


@pytest.mark.unit
def test_truncation_check_runs_only_when_cap_configured(tmp_path: Path) -> None:
    flagged_rows = [{"request_id": 1, "suspected_length_cap_truncation": 1}]

    with_cap = _health_checker(tmp_path, flagged_rows, max_expected_audio_ms=1000.0)
    check_names = [result.summary["name"] for result in with_cap.run_checks()]
    assert "Suspected Length-Cap Truncation Check" in check_names

    without_cap = _health_checker(tmp_path, flagged_rows, max_expected_audio_ms=None)
    check_names = [result.summary["name"] for result in without_cap.run_checks()]
    assert "Suspected Length-Cap Truncation Check" not in check_names
