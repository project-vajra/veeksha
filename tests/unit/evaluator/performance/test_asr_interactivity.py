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
