import pytest

from veeksha.evaluator.performance.asr import (
    ASRMetricAccumulator,
    compute_interactivity_stats,
    score_asr_request,
)


@pytest.mark.unit
def test_compute_interactivity_stats_from_snapshots() -> None:
    stats = compute_interactivity_stats(
        {
            "reference_word_timestamps": [
                {"word": "hello", "start_ms": 0, "end_ms": 100},
                {"word": "world", "start_ms": 200, "end_ms": 300},
            ],
            "transcript_snapshots": [
                {"elapsed_ms": 150, "transcript": "hello"},
                {"elapsed_ms": 320, "transcript": "hello world"},
            ],
        }
    )

    assert stats is not None
    assert stats.word_count == 2
    assert stats.latencies_ms == [50, 20]
    assert stats.mean_latency_ms == 35


@pytest.mark.unit
def test_compute_interactivity_expands_and_skips_normalized_reference_words() -> None:
    stats = compute_interactivity_stats(
        {
            "reference_word_timestamps": [
                {"word": "don't", "start_ms": 0, "end_ms": 100},
                {"word": "uh", "start_ms": 120, "end_ms": 180},
                {"word": "stop", "start_ms": 200, "end_ms": 300},
            ],
            "transcript_snapshots": [
                {"elapsed_ms": 150, "transcript": "do not"},
                {"elapsed_ms": 340, "transcript": "do not stop"},
            ],
        }
    )

    assert stats is not None
    assert stats.word_count == 3
    assert stats.latencies_ms == [50, 50, 40]
    assert stats.mean_latency_ms == pytest.approx(46.6666667)


@pytest.mark.unit
def test_compute_interactivity_stats_empty_snapshots_returns_none() -> None:
    stats = compute_interactivity_stats(
        {
            "reference_word_timestamps": [
                {"word": "hello", "start_ms": 0, "end_ms": 100},
            ],
            "transcript_snapshots": [],
        }
    )

    assert stats is None


@pytest.mark.unit
def test_score_asr_request_writes_interactivity_row_fields() -> None:
    metrics = score_asr_request(
        request_id=0,
        channel_metrics={
            "final_transcript": "hello world",
            "expected_transcript": "hello world",
            "dataset": "toy",
            "reference_word_timestamps": [
                {"word": "hello", "start_ms": 0, "end_ms": 100},
                {"word": "world", "start_ms": 200, "end_ms": 300},
            ],
            "transcript_snapshots": [
                {"elapsed_ms": 150, "transcript": "hello"},
                {"elapsed_ms": 320, "transcript": "hello world"},
            ],
        },
        duration_s=0.3,
        accumulator=ASRMetricAccumulator(),
    )

    row = metrics.to_request_row()
    assert row["interactivity"] == 35
    assert row["interactivity_word_count"] == 2
