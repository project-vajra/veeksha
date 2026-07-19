from __future__ import annotations

import pytest

from veeksha.evaluator.performance.audio_interactivity import (
    RequestTiming,
    compute_audio_fluidity,
    compute_interactivity_metrics,
)


def _chunk(receipt_ms: float, duration_ms: float) -> tuple[float, float, float]:
    # 16-bit mono PCM at 24 kHz: 48 decoded bytes per millisecond.
    return receipt_ms, duration_ms, duration_ms * 48


def test_audio_fluidity_carries_early_frame_slack_forward() -> None:
    result = compute_audio_fluidity(
        [_chunk(100.0, 60.0), _chunk(150.0, 20.0)],
        frame_duration_ms=20.0,
        startup_delay_ms=0.0,
    )

    assert result is not None
    assert result.playable_frame_count == 4
    assert result.missed_deadlines == 0
    assert result.total_deadlines == 4
    assert result.fluidity_index == 1.0


def test_audio_fluidity_counts_misses_and_resets_after_a_stall() -> None:
    chunks = [
        _chunk(100.0, 20.0),
        _chunk(145.0, 20.0),
        _chunk(165.0, 20.0),
    ]

    zero_delay = compute_audio_fluidity(
        chunks,
        frame_duration_ms=20.0,
        startup_delay_ms=0.0,
    )
    buffered = compute_audio_fluidity(
        chunks,
        frame_duration_ms=20.0,
        startup_delay_ms=25.0,
    )

    assert zero_delay is not None
    assert zero_delay.playable_frame_count == 3
    assert zero_delay.missed_deadlines == 2
    assert zero_delay.total_deadlines == 4
    assert zero_delay.fluidity_index == pytest.approx(0.5)
    assert buffered is not None
    assert buffered.fluidity_index == 1.0


def test_audio_fluidity_is_invariant_to_transport_fragmentation() -> None:
    coarse_chunks = [_chunk(100.0, 40.0), _chunk(140.0, 40.0)]
    fragmented_chunks = [
        _chunk(90.0, 10.0),
        _chunk(100.0, 30.0),
        _chunk(130.0, 10.0),
        _chunk(140.0, 30.0),
    ]

    coarse = compute_audio_fluidity(
        coarse_chunks,
        frame_duration_ms=20.0,
        startup_delay_ms=0.0,
    )
    fragmented = compute_audio_fluidity(
        fragmented_chunks,
        frame_duration_ms=20.0,
        startup_delay_ms=0.0,
    )

    assert coarse is not None
    assert fragmented is not None
    assert coarse.playable_frame_count == fragmented.playable_frame_count == 4
    assert coarse.total_deadlines == fragmented.total_deadlines
    assert coarse.missed_deadlines == fragmented.missed_deadlines
    assert coarse.fluidity_index == fragmented.fluidity_index


def test_audio_fluidity_ignores_an_incomplete_tail_frame() -> None:
    assert (
        compute_audio_fluidity(
            [_chunk(100.0, 19.0)],
            frame_duration_ms=20.0,
            startup_delay_ms=0.0,
        )
        is None
    )


def test_interactivity_distinguishes_first_byte_from_first_playable_frame() -> None:
    timing = RequestTiming(
        text_deltas=[(20.0, 5)],
        audio_chunks=[_chunk(100.0, 10.0), _chunk(110.0, 10.0)],
        commit_ms=80.0,
        audio_done_ms=120.0,
        response_done_ms=125.0,
        sample_rate=24000,
    )

    metrics = compute_interactivity_metrics(
        timing,
        startup_delay_ms_values=[0.0, 100.0],
        startup_buffer_ms_values=[0.0],
        min_reportable_stall_ms=10.0,
        fluidity_frame_ms=20.0,
        fluidity_startup_delay_ms=100.0,
        fluidity_attribution_mode="conservative",
    )

    assert metrics.request_start_to_first_audio_ms == 100.0
    assert metrics.request_start_to_first_playable_audio_ms == 110.0
    assert metrics.user_audio_fluidity is not None
    assert metrics.user_audio_fluidity.startup_delay_ms == 100.0
    assert metrics.user_audio_fluidity.fluidity_index == 1.0


def test_duplex_user_fluidity_is_not_attributed_to_service_without_source_proof() -> (
    None
):
    timing = RequestTiming(
        text_deltas=[(0.0, 5), (200.0, 5)],
        audio_chunks=[
            _chunk(50.0, 20.0),
            _chunk(115.0, 20.0),
        ],
        commit_ms=200.0,
        audio_done_ms=140.0,
        response_done_ms=210.0,
        sample_rate=24000,
    )

    metrics = compute_interactivity_metrics(
        timing,
        startup_delay_ms_values=[0.0],
        startup_buffer_ms_values=[0.0],
        min_reportable_stall_ms=10.0,
        fluidity_frame_ms=20.0,
        fluidity_startup_delay_ms=0.0,
        fluidity_attribution_mode="conservative",
    )

    assert metrics.duplex_overlap_observed
    assert metrics.user_audio_fluidity is not None
    assert metrics.user_audio_fluidity.missed_deadlines > 0
    assert metrics.tts_service_fluidity is None
    assert not metrics.tts_service_fluidity_eligible
    assert metrics.unattributed_missed_deadlines > 0


def test_oversupplied_duplex_attributes_fluidity_to_service() -> None:
    timing = RequestTiming(
        text_deltas=[(0.0, 5), (200.0, 5)],
        audio_chunks=[
            _chunk(50.0, 20.0),
            _chunk(115.0, 20.0),
        ],
        commit_ms=200.0,
        audio_done_ms=140.0,
        response_done_ms=210.0,
        sample_rate=24000,
    )

    metrics = compute_interactivity_metrics(
        timing,
        startup_delay_ms_values=[0.0],
        startup_buffer_ms_values=[0.0],
        min_reportable_stall_ms=10.0,
        fluidity_frame_ms=20.0,
        fluidity_startup_delay_ms=0.0,
        fluidity_attribution_mode="source_oversupplied",
    )

    assert metrics.tts_service_fluidity is metrics.user_audio_fluidity
    assert metrics.tts_service_fluidity_eligible
    assert metrics.unattributed_missed_deadlines == 0


@pytest.mark.parametrize(
    ("frame_ms", "startup_ms", "message"),
    [
        (0.0, 0.0, "frame_duration_ms must be > 0"),
        (20.0, -1.0, "startup_delay_ms must be >= 0"),
    ],
)
def test_audio_fluidity_rejects_invalid_policy(
    frame_ms: float, startup_ms: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_audio_fluidity(
            [_chunk(100.0, 20.0)],
            frame_duration_ms=frame_ms,
            startup_delay_ms=startup_ms,
        )
