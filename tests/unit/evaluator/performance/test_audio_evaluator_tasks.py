from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from veeksha.config.evaluator import (
    AudioChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.evaluator.performance.audio import AudioPerformanceEvaluator
from veeksha.types import AudioTask, ChannelModality


def _evaluator(
    *,
    interactivity_enabled: bool,
    max_expected_audio_ms: float | None = None,
) -> AudioPerformanceEvaluator:
    audio_config = AudioChannelPerformanceConfig(
        interactivity_enabled=interactivity_enabled,
        startup_delay_ms_values=[0.0],
        startup_buffer_ms_values=[0.0],
        max_expected_audio_ms=max_expected_audio_ms,
    )
    config = PerformanceEvaluatorConfig(
        target_channels=[ChannelModality.AUDIO],
        slos=[],
        audio_channel=audio_config,
    )
    return AudioPerformanceEvaluator(
        config,
        channel_config=audio_config,
        benchmark_start_time=100.0,
    )


def _record(
    evaluator: AudioPerformanceEvaluator,
    *,
    request_id: int,
    content: bytes | str,
    metrics: dict,
) -> None:
    evaluator.register_request(
        request_id=request_id,
        session_id=request_id,
        dispatched_at=100.0,
        content=None,
    )
    evaluator.record_request_completed(
        request_id=request_id,
        session_id=request_id,
        completed_at=101.0,
        response=RequestResult(
            request_id=request_id,
            session_id=request_id,
            channels={
                ChannelModality.AUDIO: ChannelResponse(
                    modality=ChannelModality.AUDIO,
                    content=content,
                    metrics=metrics,
                )
            },
            scheduler_dispatched_at=100.0,
            client_completed_at=101.0,
        ),
    )


def _wav_bytes(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def test_tts_task_preserves_playback_interactivity_metrics() -> None:
    evaluator = _evaluator(interactivity_enabled=True)

    _record(
        evaluator,
        request_id=1,
        content=b"\0" * 4800,
        metrics={
            "audio_task": AudioTask.TTS,
            AudioMetricKey.TTFC.value: 100.0,
            AudioMetricKey.END_TO_END_LATENCY.value: 250.0,
            AudioMetricKey.CHUNK_COUNT.value: 2,
            AudioMetricKey.RAW_PCM.value: True,
            AudioMetricKey.SAMPLE_RATE.value: 24000,
            AudioMetricKey.INPUT_CHARS.value: 5,
            AudioMetricKey.INPUT_TOKENS.value: 1,
            AudioMetricKey.INPUT_TEXT.value: "hello",
            AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value: [[20.0, 5]],
            AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value: [
                [100.0, 2400],
                [180.0, 2400],
            ],
            AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value: 150.0,
            AudioMetricKey.AUDIO_DONE_OFFSET_MS.value: 230.0,
            AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value: 240.0,
        },
    )

    row = evaluator._export_request_rows()[0]
    assert row["audio_task"] == "tts"
    assert row[AudioMetricKey.GENERATED_AUDIO_DURATION.value] == 100.0
    assert row[AudioMetricKey.ZERO_DELAY_STALL_COUNT.value] == 1
    assert row[AudioMetricKey.INPUT_TEXT.value] == "hello"


def test_stt_task_uses_input_pcm_duration_and_exports_asr_metrics() -> None:
    evaluator = _evaluator(interactivity_enabled=True)

    _record(
        evaluator,
        request_id=2,
        content="hello world",
        metrics={
            "audio_task": "stt",
            AudioMetricKey.TTFC.value: 120.0,
            AudioMetricKey.END_TO_END_LATENCY.value: 1000.0,
            AudioMetricKey.CHUNK_COUNT.value: 4,
            AudioMetricKey.PCM_BYTE_COUNT.value: 32000,
            AudioMetricKey.RAW_PCM.value: True,
            AudioMetricKey.SAMPLE_RATE.value: 16000,
            "final_transcript": "hello world",
            "partial_transcript": "hello",
            "expected_transcript": "hello world",
            "dataset": "fixture",
            "time_to_first_visible_text": 120.0,
            "time_to_first_partial": 150.0,
            "time_to_final_transcript": 900.0,
            "transcript_snapshots": [],
        },
    )

    row = evaluator._export_request_rows()[0]
    summary = evaluator.get_summary()
    assert row["audio_task"] == "stt"
    assert row[AudioMetricKey.PCM_BYTE_COUNT.value] == 32000
    assert row[AudioMetricKey.GENERATED_AUDIO_DURATION.value] == 1000.0
    assert row["final_wer"] == 0.0
    assert summary["asr_final_sample_count"] == 1.0
    assert AudioMetricKey.ZERO_DELAY_STALL_COUNT.value not in row


def _tts_metrics() -> dict:
    return {
        "audio_task": AudioTask.TTS,
        AudioMetricKey.TTFC.value: 100.0,
        AudioMetricKey.END_TO_END_LATENCY.value: 900.0,
        AudioMetricKey.CHUNK_COUNT.value: 1,
        AudioMetricKey.RAW_PCM.value: True,
        AudioMetricKey.SAMPLE_RATE.value: 24000,
    }


def test_duration_at_cap_is_counted_as_suspected_truncation() -> None:
    # Cap 1000ms with the 320ms one-chunk tolerance: >= 680ms is suspect.
    # At 24kHz 16-bit mono PCM, 1ms = 48 bytes.
    evaluator = _evaluator(interactivity_enabled=False, max_expected_audio_ms=1000.0)

    _record(
        evaluator,
        request_id=1,
        content=b"\0" * (1000 * 48),  # exactly at the cap
        metrics=_tts_metrics(),
    )
    _record(
        evaluator,
        request_id=2,
        content=b"\0" * (680 * 48),  # exactly at cap - one chunk
        metrics=_tts_metrics(),
    )
    _record(
        evaluator,
        request_id=3,
        content=b"\0" * (679 * 48),  # just below the threshold
        metrics=_tts_metrics(),
    )

    summary = evaluator.get_summary()
    assert summary["suspected_length_cap_truncation"] == 2

    rows = {row["request_id"]: row for row in evaluator._export_request_rows()}
    assert rows[1]["suspected_length_cap_truncation"] == 1
    assert rows[2]["suspected_length_cap_truncation"] == 1
    assert rows[3]["suspected_length_cap_truncation"] == 0


def test_truncation_detection_disabled_without_configured_cap() -> None:
    evaluator = _evaluator(interactivity_enabled=False)

    _record(
        evaluator,
        request_id=1,
        content=b"\0" * (1000 * 48),
        metrics=_tts_metrics(),
    )

    assert "suspected_length_cap_truncation" not in evaluator.get_summary()
    assert "suspected_length_cap_truncation" not in evaluator._export_request_rows()[0]


def test_audio_channel_config_rejects_nonpositive_cap() -> None:
    with pytest.raises(ValueError, match="max_expected_audio_ms must be > 0"):
        AudioChannelPerformanceConfig(max_expected_audio_ms=0.0)


def _tts_interactive_metrics(*, aborted: bool) -> dict:
    metrics = {
        "audio_task": AudioTask.TTS,
        AudioMetricKey.TTFC.value: 100.0,
        AudioMetricKey.END_TO_END_LATENCY.value: 250.0,
        AudioMetricKey.CHUNK_COUNT.value: 2,
        AudioMetricKey.RAW_PCM.value: True,
        AudioMetricKey.SAMPLE_RATE.value: 24000,
        AudioMetricKey.INPUT_CHARS.value: 5,
        AudioMetricKey.INPUT_TOKENS.value: 1,
        AudioMetricKey.INPUT_TEXT.value: "hello",
        AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value: [[20.0, 5]],
        AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value: [[100.0, 2400], [180.0, 2400]],
        AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value: 150.0,
        AudioMetricKey.AUDIO_DONE_OFFSET_MS.value: 230.0,
        AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value: 240.0,
    }
    if aborted:
        metrics[AudioMetricKey.ABORTED.value] = True
    return metrics


def test_aborted_request_bucketed_and_excluded_from_aggregates() -> None:
    evaluator = _evaluator(interactivity_enabled=True)

    _record(
        evaluator,
        request_id=1,
        content=b"\0" * 4800,
        metrics=_tts_interactive_metrics(aborted=False),
    )
    _record(
        evaluator,
        request_id=2,
        content=b"\0" * 4800,
        metrics=_tts_interactive_metrics(aborted=True),
    )

    summary = evaluator.get_summary()
    assert summary["aborted_requests_count"] == 1

    # Only the non-aborted request feeds the continuity / duration sketches and
    # the interactivity bucket.
    assert len(evaluator.summaries[AudioMetricKey.GENERATED_AUDIO_DURATION.value]) == 1
    assert len(evaluator.summaries[AudioMetricKey.TTFC.value]) == 1
    assert evaluator._interactive_request_count == 1

    rows = {row["request_id"]: row for row in evaluator._export_request_rows()}
    # Both requests are still exported; the aborted one is flagged and carries
    # no interactivity fields.
    assert rows[1][AudioMetricKey.ABORTED.value] == 0
    assert rows[2][AudioMetricKey.ABORTED.value] == 1
    assert AudioMetricKey.ZERO_DELAY_STALL_COUNT.value in rows[1]
    assert AudioMetricKey.ZERO_DELAY_STALL_COUNT.value not in rows[2]


def test_aborted_request_not_flagged_as_length_cap_truncation() -> None:
    # Cap 1000ms: >= 680ms is suspect. An aborted stream at/over that duration
    # is partial by construction and must not be counted as truncation.
    evaluator = _evaluator(interactivity_enabled=False, max_expected_audio_ms=1000.0)

    _record(
        evaluator,
        request_id=1,
        content=b"\0" * (1000 * 48),
        metrics={**_tts_metrics(), AudioMetricKey.ABORTED.value: True},
    )

    summary = evaluator.get_summary()
    assert summary["suspected_length_cap_truncation"] == 0
    assert summary["aborted_requests_count"] == 1
    row = evaluator._export_request_rows()[0]
    assert row["suspected_length_cap_truncation"] == 0
    assert row[AudioMetricKey.ABORTED.value] == 1


def test_summary_has_no_aborted_count_without_aborts() -> None:
    evaluator = _evaluator(interactivity_enabled=False)
    _record(evaluator, request_id=1, content=b"\0" * 4800, metrics=_tts_metrics())
    assert "aborted_requests_count" not in evaluator.get_summary()


def test_audio_response_without_task_is_rejected() -> None:
    evaluator = _evaluator(interactivity_enabled=False)

    with pytest.raises(ValueError, match="unknown audio_task"):
        _record(
            evaluator,
            request_id=3,
            content=b"\0" * 320,
            metrics={
                AudioMetricKey.RAW_PCM.value: True,
                AudioMetricKey.SAMPLE_RATE.value: 16000,
            },
        )


def test_audio_integrity_raw_and_wav_use_same_final_pcm16_boundary() -> None:
    evaluator = _evaluator(interactivity_enabled=False)
    clean_pcm = np.tile(
        np.asarray([0, 8192, -8192, 16384, -16384], dtype="<i2"), 16
    ).tobytes()

    _record(
        evaluator,
        request_id=1,
        content=clean_pcm,
        metrics=_tts_metrics(),
    )
    _record(
        evaluator,
        request_id=2,
        content=_wav_bytes(clean_pcm, sample_rate=22050),
        metrics={
            **_tts_metrics(),
            AudioMetricKey.RAW_PCM.value: False,
        },
    )

    rows = {row["request_id"]: row for row in evaluator._export_request_rows()}
    for metric_name in (
        AudioMetricKey.PCM_SAMPLE_COUNT.value,
        AudioMetricKey.PEAK_ABS_AMPLITUDE.value,
        AudioMetricKey.CLIPPED_SAMPLE_FRACTION.value,
        AudioMetricKey.RMS.value,
        AudioMetricKey.NON_FINITE_SAMPLE_COUNT.value,
        AudioMetricKey.AUDIO_SUSPECT.value,
    ):
        assert rows[1][metric_name] == rows[2][metric_name]
    assert rows[1][AudioMetricKey.AUDIO_SUSPECT.value] == 0

    summary = evaluator.get_summary()
    assert summary[AudioMetricKey.AUDIO_SUSPECT.value] is False
    assert summary["audio_integrity_requests_count"] == 2
    assert summary["audio_suspect_requests_count"] == 0
    assert summary["audio_suspect_requests_fraction"] == 0.0


def test_audio_integrity_clipped_request_sets_request_and_summary_gate() -> None:
    evaluator = _evaluator(interactivity_enabled=False)
    clipped_pcm = np.tile(np.asarray([32767, -32768, 0, 0], dtype="<i2"), 16).tobytes()

    _record(
        evaluator,
        request_id=1,
        content=clipped_pcm,
        metrics=_tts_metrics(),
    )

    row = evaluator._export_request_rows()[0]
    assert row[AudioMetricKey.CLIPPED_SAMPLE_FRACTION.value] == 0.5
    assert row[AudioMetricKey.AUDIO_SUSPECT.value] == 1

    summary = evaluator.get_summary()
    assert summary[AudioMetricKey.AUDIO_SUSPECT.value] is True
    assert summary["audio_suspect_requests_count"] == 1
    assert summary["audio_suspect_requests_fraction"] == 1.0


def test_audio_integrity_failed_request_keeps_metrics_but_not_summary_gate() -> None:
    evaluator = _evaluator(interactivity_enabled=False)
    evaluator.register_request(
        request_id=1,
        session_id=1,
        dispatched_at=100.0,
        content=None,
    )
    evaluator.record_request_completed(
        request_id=1,
        session_id=1,
        completed_at=101.0,
        response=RequestResult(
            request_id=1,
            session_id=1,
            success=False,
            channels={
                ChannelModality.AUDIO: ChannelResponse(
                    modality=ChannelModality.AUDIO,
                    content=b"\0\0",
                    metrics=_tts_metrics(),
                )
            },
        ),
    )

    row = evaluator._export_request_rows()[0]
    assert row[AudioMetricKey.PCM_SAMPLE_COUNT.value] == 1
    assert row[AudioMetricKey.AUDIO_SUSPECT.value] == 0
    assert evaluator.get_summary()["audio_integrity_requests_count"] == 0


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("audio_integrity_min_peak_abs_amplitude", -0.1, "must be >= 0"),
        ("audio_integrity_max_clipped_sample_fraction", 1.1, "must be in"),
        ("audio_integrity_min_rms", -0.1, "must be >= 0"),
        ("audio_integrity_max_rms", 0.0, "must be > 0"),
        ("audio_integrity_max_non_finite_sample_count", -1, "must be >= 0"),
    ],
)
def test_audio_integrity_config_rejects_invalid_thresholds(
    field_name: str, value: float | int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AudioChannelPerformanceConfig(**{field_name: value})
