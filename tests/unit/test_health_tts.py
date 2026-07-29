"""Tests for TTS-specific health checks (truncation, zombie sessions)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import veeksha.health as health_module
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.endpoint import EndpointConfig
from veeksha.config.evaluator import (
    AudioChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.config.traffic import ConcurrentTrafficConfig
from veeksha.health import (
    HealthChecker,
    TTSWorkerStatsSnapshot,
    TTSZombieSessionProbe,
    maybe_build_tts_zombie_probe,
)


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


# ---------------------------------------------------------------------------
# Session concurrency check
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_concurrency_check_uses_client_completion_for_audio_latency(
    tmp_path: Path,
) -> None:
    metrics_file = tmp_path / "request_level_metrics.jsonl"
    _write_metrics(
        metrics_file,
        [
            {
                "request_id": 0,
                "session_id": 0,
                "session_size": 1,
                "scheduler_dispatched_at": 0.0,
                "client_completed_at": 1.0,
                "end_to_end_latency": 1000.0,
            },
            {
                "request_id": 1,
                "session_id": 1,
                "session_size": 1,
                "scheduler_dispatched_at": 1.1,
                "client_completed_at": 2.0,
                "end_to_end_latency": 900.0,
            },
        ],
    )
    checker = HealthChecker(
        trace_file=str(tmp_path / "missing_trace.jsonl"),
        metrics_file=str(metrics_file),
        benchmark_config=BenchmarkConfig(
            traffic_scheduler=ConcurrentTrafficConfig(
                target_concurrent_sessions=1,
                rampup_seconds=0,
            ),
            evaluators=_benchmark_config(max_expected_audio_ms=None).evaluators,
        ),
    )
    assert checker.load_data()

    result = checker.check_session_concurrency()

    assert result.passed is True
    overall_results = next(
        section["results"]
        for section in result.summary["sections"]
        if section["title"] == "Overall Statistics"
    )
    assert overall_results["Max Observed Concurrency"] == "1"


# ---------------------------------------------------------------------------
# TTS zombie-session probe
# ---------------------------------------------------------------------------


def _probe_with_snapshots(
    start: TTSWorkerStatsSnapshot | None,
    end: TTSWorkerStatsSnapshot | None,
) -> TTSZombieSessionProbe:
    probe = TTSZombieSessionProbe("http://localhost:8081")
    probe._start_snapshot = start
    probe._end_snapshot = end
    return probe


@pytest.mark.unit
def test_zombie_probe_fails_on_finished_session_surplus() -> None:
    probe = _probe_with_snapshots(
        TTSWorkerStatsSnapshot(finished_eos=100, finished_length_cap=5),
        TTSWorkerStatsSnapshot(finished_eos=400, finished_length_cap=27),
    )

    result = probe.build_result(completed_requests=200)

    assert result.passed is False
    results = result.summary["sections"][0]["results"]
    assert results["Server Finished Delta"] == "322 (eos: 300, length_cap: 22)"
    assert results["Surplus (zombies)"] == "122"
    assert "Interpretation" in results


@pytest.mark.unit
def test_zombie_probe_passes_when_delta_matches_completions() -> None:
    probe = _probe_with_snapshots(
        TTSWorkerStatsSnapshot(finished_eos=10, finished_length_cap=0),
        TTSWorkerStatsSnapshot(finished_eos=210, finished_length_cap=0),
    )

    result = probe.build_result(completed_requests=200)

    assert result.passed is True


@pytest.mark.unit
def test_zombie_probe_tolerates_deficit_with_note() -> None:
    # Fewer server-side finishes than completions: sessions still decoding at
    # the end snapshot. Reported but not a failure.
    probe = _probe_with_snapshots(
        TTSWorkerStatsSnapshot(finished_eos=0, finished_length_cap=0),
        TTSWorkerStatsSnapshot(finished_eos=150, finished_length_cap=0),
    )

    result = probe.build_result(completed_requests=200)

    assert result.passed is True
    assert "Note" in result.summary["sections"][0]["results"]


@pytest.mark.unit
def test_zombie_probe_skips_when_snapshot_unavailable() -> None:
    probe = _probe_with_snapshots(
        None, TTSWorkerStatsSnapshot(finished_eos=1, finished_length_cap=0)
    )
    probe._start_note = "GET http://localhost:8081/debug/tts_worker_stats failed"

    result = probe.build_result(completed_requests=5)

    assert result.passed is True
    results = result.summary["sections"][0]["results"]
    assert results["Status"] == "Skipped"
    assert "failed" in results["Reason"]


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


@pytest.mark.unit
def test_zombie_probe_fetch_parses_talker_finished_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "talker": {"finished": {"eos": 42, "length_cap": 7}, "steps": 999},
        "code2wav": {},
    }
    monkeypatch.setattr(
        health_module.requests,
        "get",
        lambda url, timeout: _FakeResponse(200, payload),
    )
    probe = TTSZombieSessionProbe("http://localhost:8081")

    probe.capture_start()

    assert probe._start_note is None
    assert probe._start_snapshot == TTSWorkerStatsSnapshot(
        finished_eos=42, finished_length_cap=7
    )


@pytest.mark.unit
def test_zombie_probe_fetch_skips_on_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        health_module.requests,
        "get",
        lambda url, timeout: _FakeResponse(404),
    )
    probe = TTSZombieSessionProbe("http://localhost:8081")

    probe.capture_start()
    probe.capture_end()

    assert probe._start_snapshot is None
    assert "404" in probe._start_note
    # A skipped probe must never fail the run.
    assert probe.build_result(completed_requests=0).passed is True


@pytest.mark.unit
def test_zombie_probe_fetch_notes_missing_talker_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        health_module.requests,
        "get",
        lambda url, timeout: _FakeResponse(200, {"code2wav": {}}),
    )
    probe = TTSZombieSessionProbe("http://localhost:8081")

    probe.capture_start()

    assert probe._start_snapshot is None
    assert "talker.finished" in probe._start_note


def _endpoint(engine_type: str, health_url: str | None) -> EndpointConfig:
    return EndpointConfig(
        engine_type=engine_type,
        model="qwen-tts",
        api_base="http://localhost:8081",
        health_url=health_url,
    )


@pytest.mark.unit
def test_maybe_build_probe_requires_vajra_endpoint_with_health_url() -> None:
    vajra = BenchmarkConfig(endpoint=_endpoint("vajra", "http://localhost:8081/health"))
    probe = maybe_build_tts_zombie_probe(vajra)
    assert probe is not None
    assert probe.stats_url == "http://localhost:8081/debug/tts_worker_stats"

    no_health_url = BenchmarkConfig(endpoint=_endpoint("vajra", None))
    assert maybe_build_tts_zombie_probe(no_health_url) is None

    non_vajra = BenchmarkConfig(
        endpoint=_endpoint("vllm", "http://localhost:8081/health")
    )
    assert maybe_build_tts_zombie_probe(non_vajra) is None

    no_endpoint = BenchmarkConfig()
    assert maybe_build_tts_zombie_probe(no_endpoint) is None


@pytest.mark.unit
def test_zombie_check_included_in_run_checks_with_probe(tmp_path: Path) -> None:
    rows = [
        {"request_id": 1, "suspected_length_cap_truncation": 0},
        {"request_id": 2, "suspected_length_cap_truncation": 0},
    ]
    metrics_file = tmp_path / "request_level_metrics.jsonl"
    _write_metrics(metrics_file, rows)
    probe = _probe_with_snapshots(
        TTSWorkerStatsSnapshot(finished_eos=0, finished_length_cap=0),
        TTSWorkerStatsSnapshot(finished_eos=2, finished_length_cap=0),
    )
    checker = HealthChecker(
        trace_file=str(tmp_path / "missing_trace.jsonl"),
        metrics_file=str(metrics_file),
        benchmark_config=_benchmark_config(max_expected_audio_ms=None),
        tts_zombie_probe=probe,
    )
    assert checker.load_data()

    results = {result.summary["name"]: result for result in checker.run_checks()}

    # Delta (2) == completed rows (2): passes.
    assert results["TTS Zombie Session Check"].passed is True
