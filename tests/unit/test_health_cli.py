"""Tests for the standalone `veeksha health` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.evaluator import (
    AudioChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.config.health_check import HealthCheckConfig
from veeksha.config.utils import to_serializable_config_dict
from veeksha.health import (
    _extract_check_block,
    load_benchmark_config_from_run_dir,
    run_health_check,
    run_health_check_cli,
)

_ZOMBIE_BLOCK = (
    "============================================================\n"
    "TTS ZOMBIE SESSION CHECK\n"
    "============================================================\n"
    "Result: FAILED\n"
    "\n"
    "Finished Session Accounting:\n"
    "  Surplus (zombies)              122\n"
)


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


def _make_run_dir(
    tmp_path: Path,
    rows: list[dict],
    max_expected_audio_ms: float | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "metrics").mkdir(parents=True)
    config_dict = to_serializable_config_dict(_benchmark_config(max_expected_audio_ms))
    (run_dir / "config.yml").write_text(yaml.safe_dump(config_dict, sort_keys=False))
    with open(run_dir / "metrics" / "request_level_metrics.jsonl", "w") as file:
        for row in rows:
            full_row = {
                "session_id": row["request_id"],
                "scheduler_dispatched_at": float(row["request_id"]),
                "client_completed_at": float(row["request_id"]) + 1.0,
                **row,
            }
            file.write(json.dumps(full_row) + "\n")
    return run_dir


@pytest.mark.unit
def test_config_round_trips_from_run_dir(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path, rows=[{"request_id": 1}], max_expected_audio_ms=163840.0
    )

    config = load_benchmark_config_from_run_dir(str(run_dir))

    assert isinstance(config, BenchmarkConfig)
    assert config.evaluators[0].audio_channel.max_expected_audio_ms == 163840.0


@pytest.mark.unit
def test_run_health_check_writes_report(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path,
        rows=[
            {"request_id": 1, "suspected_length_cap_truncation": 0},
            {"request_id": 2, "suspected_length_cap_truncation": 0},
        ],
        max_expected_audio_ms=163840.0,
    )

    results = run_health_check(str(run_dir))

    report_path = run_dir / "health_check_results.txt"
    assert report_path.is_file()
    check_names = [result.summary["name"] for result in results]
    assert "Suspected Length-Cap Truncation Check" in check_names
    report_text = report_path.read_text()
    assert "SUSPECTED LENGTH-CAP TRUNCATION CHECK" in report_text


@pytest.mark.unit
def test_run_health_check_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config.yml"):
        run_health_check(str(tmp_path))


@pytest.mark.unit
def test_run_health_check_carries_over_zombie_block(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, rows=[{"request_id": 1}])
    report_path = run_dir / "health_check_results.txt"
    report_path.write_text(_ZOMBIE_BLOCK)

    results = run_health_check(str(run_dir))

    report_text = report_path.read_text()
    assert "TTS ZOMBIE SESSION CHECK" in report_text
    assert "carried over from the in-run report" in report_text
    carried = next(
        result
        for result in results
        if result.summary["name"] == "TTS Zombie Session Check (carried over)"
    )
    assert carried.passed is False

    # Re-running against the regenerated report must not duplicate the note.
    run_health_check(str(run_dir))
    assert report_path.read_text().count("carried over from the in-run report") == 1


@pytest.mark.unit
def test_extract_check_block_returns_none_when_absent() -> None:
    assert _extract_check_block("no blocks here", "TTS Zombie Session Check") is None


@pytest.mark.unit
def test_cli_strict_exits_nonzero_on_failure(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path,
        rows=[{"request_id": 1, "suspected_length_cap_truncation": 1}],
        max_expected_audio_ms=163840.0,
    )
    strict = HealthCheckConfig(run_dir=str(run_dir))
    assert strict.strict is True

    with pytest.raises(SystemExit) as exc_info:
        run_health_check_cli([strict])
    assert exc_info.value.code == 1

    # Non-strict mode reports but does not fail the process.
    run_health_check_cli([HealthCheckConfig(run_dir=str(run_dir), strict=False)])


@pytest.mark.unit
def test_cli_respects_output_file_override(tmp_path: Path) -> None:
    run_dir = _make_run_dir(
        tmp_path,
        rows=[{"request_id": 1, "suspected_length_cap_truncation": 0}],
        max_expected_audio_ms=163840.0,
    )
    override = tmp_path / "custom_report.txt"

    run_health_check_cli(
        [HealthCheckConfig(run_dir=str(run_dir), output_file=str(override))]
    )

    assert override.is_file()
    assert not (run_dir / "health_check_results.txt").exists()


@pytest.mark.unit
def test_health_requires_run_dir() -> None:
    with pytest.raises(ValueError, match="--run_dir"):
        HealthCheckConfig()
