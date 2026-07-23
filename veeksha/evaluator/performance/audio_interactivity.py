"""Pure computation for realtime TTS interactivity metrics.

The websocket client records one monotonic timeline per request. This module
replays that timeline locally under multiple playback policies; no additional
server requests are made for a policy sweep.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from veeksha.core.audio_contract import (
    DEFAULT_AUDIO_SAMPLE_RATE,
    AudioMetricKey,
    pcm_bytes_to_duration_ms,
)


@dataclass(frozen=True)
class RequestTiming:
    """Validated realtime request timing.

    Audio tuples contain ``(receipt_offset_ms, duration_ms, decoded_bytes)``.
    Chunks received at the same timestamp are coalesced so playback metrics do
    not depend on transport frame fragmentation.
    """

    text_deltas: list[tuple[float, int]]
    audio_chunks: list[tuple[float, float, float]]
    response_trigger_ms: Optional[float]
    commit_ms: Optional[float]
    audio_done_ms: Optional[float]
    response_done_ms: Optional[float]
    sample_rate: int


def _optional_float(metrics: Mapping, key: AudioMetricKey) -> Optional[float]:
    value = metrics.get(key.value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_request_timing(
    metrics: Mapping, default_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
) -> Optional[RequestTiming]:
    """Parse the raw client contract, returning ``None`` without usable audio."""
    raw_chunks = metrics.get(AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value)
    if not isinstance(raw_chunks, Sequence) or not raw_chunks:
        return None

    try:
        sample_rate = int(
            metrics.get(AudioMetricKey.SAMPLE_RATE.value, default_sample_rate)
            or default_sample_rate
        )
    except (TypeError, ValueError):
        sample_rate = default_sample_rate
    if sample_rate <= 0:
        sample_rate = default_sample_rate

    parsed_chunks: list[tuple[float, float]] = []
    for entry in raw_chunks:
        if not isinstance(entry, Sequence) or len(entry) < 2:
            continue
        try:
            offset_ms = float(entry[0])
            n_bytes = float(entry[1])
        except (TypeError, ValueError):
            continue
        if offset_ms < 0 or n_bytes <= 0:
            continue
        parsed_chunks.append((offset_ms, n_bytes))
    if not parsed_chunks:
        return None
    parsed_chunks.sort(key=lambda item: item[0])

    audio_chunks: list[tuple[float, float, float]] = []
    for offset_ms, n_bytes in parsed_chunks:
        if audio_chunks and audio_chunks[-1][0] == offset_ms:
            _, previous_duration, previous_bytes = audio_chunks[-1]
            audio_chunks[-1] = (
                offset_ms,
                previous_duration + pcm_bytes_to_duration_ms(n_bytes, sample_rate),
                previous_bytes + n_bytes,
            )
        else:
            audio_chunks.append(
                (
                    offset_ms,
                    pcm_bytes_to_duration_ms(n_bytes, sample_rate),
                    n_bytes,
                )
            )

    text_deltas: list[tuple[float, int]] = []
    raw_text_deltas = metrics.get(AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value) or []
    if isinstance(raw_text_deltas, Sequence):
        for entry in raw_text_deltas:
            if not isinstance(entry, Sequence) or len(entry) < 2:
                continue
            try:
                offset_ms = float(entry[0])
                n_chars = int(entry[1])
            except (TypeError, ValueError):
                continue
            if offset_ms >= 0 and n_chars >= 0:
                text_deltas.append((offset_ms, n_chars))
    text_deltas.sort(key=lambda item: item[0])

    return RequestTiming(
        text_deltas=text_deltas,
        audio_chunks=audio_chunks,
        response_trigger_ms=_optional_float(
            metrics, AudioMetricKey.RESPONSE_TRIGGER_OFFSET_MS
        ),
        commit_ms=_optional_float(metrics, AudioMetricKey.INPUT_COMMIT_OFFSET_MS),
        audio_done_ms=_optional_float(metrics, AudioMetricKey.AUDIO_DONE_OFFSET_MS),
        response_done_ms=_optional_float(
            metrics, AudioMetricKey.RESPONSE_DONE_OFFSET_MS
        ),
        sample_rate=sample_rate,
    )


@dataclass(frozen=True)
class PlaybackSimResult:
    """Playback outcome for one startup policy."""

    playback_start_ms: float
    startup_wait_from_first_audio_ms: float
    stall_count: int
    total_stall_ms: float
    longest_stall_ms: float
    stall_free: bool
    buffer_margin_min_ms: float
    buffer_margin_mean_ms: float


@dataclass(frozen=True)
class AudioFluidityResult:
    """Etalon-style deadline acceptance for fixed-duration playable audio.

    Network chunks are first converted into complete ``frame_duration_ms`` PCM
    frames.  A frame becomes available when cumulative received audio crosses
    the next frame boundary.  This makes the score independent of how a
    transport fragments the same playable-audio supply.

    ``startup_delay_ms`` is playback slack after the first complete frame is
    available.  Request-to-first-playable-frame latency is reported separately,
    so this score isolates continuity after audio can begin.
    """

    startup_delay_ms: float
    frame_duration_ms: float
    playable_frame_count: int
    total_deadlines: int
    missed_deadlines: int
    fluidity_index: float


def _playable_frame_arrival_ms(
    chunks: Sequence[tuple[float, float, float]],
    frame_duration_ms: float,
) -> list[float]:
    """Return receipt times for complete fixed-duration PCM frames.

    A partial tail shorter than one frame is deliberately excluded: it cannot
    fill another complete playback period and counting it as a full frame would
    over-penalize normal end-of-utterance delivery.
    """
    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be > 0")

    arrivals: list[float] = []
    cumulative_audio_ms = 0.0
    next_frame_boundary_ms = frame_duration_ms
    epsilon = frame_duration_ms * 1e-9

    for receipt_ms, duration_ms, _ in chunks:
        cumulative_audio_ms += duration_ms
        while cumulative_audio_ms + epsilon >= next_frame_boundary_ms:
            arrivals.append(receipt_ms)
            next_frame_boundary_ms += frame_duration_ms
    return arrivals


def compute_audio_fluidity(
    chunks: Sequence[tuple[float, float, float]],
    *,
    frame_duration_ms: float,
    startup_delay_ms: float,
) -> Optional[AudioFluidityResult]:
    """Compute the Etalon fluidity-index analogue for playable audio frames.

    The algorithm follows Etalon's deadline/slack/reset semantics.  Frames that
    arrive early accumulate slack (playback buffer).  When an inter-frame gap
    exceeds the frame deadline plus available slack, every elapsed playback
    deadline is counted as missed and slack is reset.
    """
    if startup_delay_ms < 0:
        raise ValueError("startup_delay_ms must be >= 0")

    arrivals = _playable_frame_arrival_ms(chunks, frame_duration_ms)
    if not arrivals:
        return None

    inter_frame_times_ms = [0.0]
    inter_frame_times_ms.extend(
        current - previous for previous, current in zip(arrivals, arrivals[1:])
    )

    total_deadlines = 0
    missed_deadlines = 0
    slack_ms = 0.0

    for index, inter_frame_ms in enumerate(inter_frame_times_ms):
        deadline_ms = startup_delay_ms if index == 0 else frame_duration_ms
        if inter_frame_ms <= deadline_ms + slack_ms:
            slack_ms += deadline_ms - inter_frame_ms
            total_deadlines += 1
            continue

        misses = (
            math.floor((inter_frame_ms - slack_ms - deadline_ms) / frame_duration_ms)
            + 1
        )
        missed_deadlines += misses
        total_deadlines += misses
        slack_ms = 0.0

    return AudioFluidityResult(
        startup_delay_ms=startup_delay_ms,
        frame_duration_ms=frame_duration_ms,
        playable_frame_count=len(arrivals),
        total_deadlines=total_deadlines,
        missed_deadlines=missed_deadlines,
        fluidity_index=(total_deadlines - missed_deadlines) / total_deadlines,
    )


def _simulate_from_start(
    chunks: Sequence[tuple[float, float, float]],
    playback_start_ms: float,
    min_reportable_stall_ms: float,
) -> PlaybackSimResult:
    """Drain delivered audio from an explicit wall-clock playback start."""
    finish_ms = playback_start_ms
    stalls: list[float] = []
    margins: list[float] = []

    for index, (receipt_ms, duration_ms, _) in enumerate(chunks):
        if index > 0:
            margins.append(finish_ms - receipt_ms)
        gap_ms = receipt_ms - finish_ms
        if index > 0 and gap_ms > min_reportable_stall_ms:
            stalls.append(gap_ms)
        finish_ms = max(finish_ms, receipt_ms) + duration_ms

    first_audio_ms = chunks[0][0]
    return PlaybackSimResult(
        playback_start_ms=playback_start_ms,
        startup_wait_from_first_audio_ms=playback_start_ms - first_audio_ms,
        stall_count=len(stalls),
        total_stall_ms=float(sum(stalls)),
        longest_stall_ms=float(max(stalls, default=0.0)),
        stall_free=not stalls,
        buffer_margin_min_ms=float(min(margins, default=0.0)),
        buffer_margin_mean_ms=(float(sum(margins) / len(margins)) if margins else 0.0),
    )


def simulate_fixed_delay(
    chunks: Sequence[tuple[float, float, float]],
    startup_delay_ms: float,
    min_reportable_stall_ms: float,
) -> Optional[PlaybackSimResult]:
    """Start playback a fixed delay after the first audio receipt."""
    if not chunks:
        return None
    return _simulate_from_start(
        chunks,
        chunks[0][0] + startup_delay_ms,
        min_reportable_stall_ms,
    )


def simulate_buffer_target(
    chunks: Sequence[tuple[float, float, float]],
    target_audio_ms: float,
    min_reportable_stall_ms: float,
    *,
    audio_done_ms: Optional[float],
    response_done_ms: Optional[float],
) -> Optional[PlaybackSimResult]:
    """Start once a target duration is buffered, or at a terminal event.

    Falling back to a terminal event lets short, complete utterances play even
    when their total duration is below the configured target. Unterminated
    partial streams remain ineligible for a target they never reached.
    """
    if not chunks:
        return None

    playback_start_ms: Optional[float] = None
    buffered_ms = 0.0
    for receipt_ms, duration_ms, _ in chunks:
        buffered_ms += duration_ms
        if target_audio_ms <= 0 or buffered_ms >= target_audio_ms:
            playback_start_ms = receipt_ms
            break

    if playback_start_ms is None:
        terminal_ms = audio_done_ms
        if terminal_ms is None:
            terminal_ms = response_done_ms
        if terminal_ms is None:
            return None
        playback_start_ms = max(terminal_ms, chunks[-1][0])

    return _simulate_from_start(chunks, playback_start_ms, min_reportable_stall_ms)


def _required_startup_delay_ms(
    chunks: Sequence[tuple[float, float, float]],
) -> float:
    """Return the minimum exact fixed delay that prevents every underrun."""
    first_audio_ms = chunks[0][0]
    cumulative_previous_ms = 0.0
    worst_deficit_ms = 0.0
    for index in range(1, len(chunks)):
        cumulative_previous_ms += chunks[index - 1][1]
        deficit_ms = chunks[index][0] - first_audio_ms - cumulative_previous_ms
        worst_deficit_ms = max(worst_deficit_ms, deficit_ms)
    return max(0.0, worst_deficit_ms)


@dataclass(frozen=True)
class InteractivityMetrics:
    """Stable and diagnostic metrics derived from one request timeline."""

    first_input_to_first_audio_ms: Optional[float]
    first_input_to_first_playable_audio_ms: Optional[float]
    trigger_to_first_playable_audio_ms: Optional[float]
    request_start_to_first_audio_ms: float
    request_start_to_first_playable_audio_ms: Optional[float]
    audio_before_commit_ratio: Optional[float]
    duplex_overlap_observed: bool
    duplex_overlap_ms: float
    post_commit_audio_delivery_ms: Optional[float]
    required_startup_delay_ms: float
    fixed_delay_playback: dict[float, PlaybackSimResult]
    buffer_target_playback: dict[float, Optional[PlaybackSimResult]]
    streaming_rtf: Optional[float]
    done_after_last_audio_ms: Optional[float]
    user_audio_fluidity: Optional[AudioFluidityResult]
    tts_service_fluidity: Optional[AudioFluidityResult]
    tts_service_fluidity_eligible: bool
    unattributed_missed_deadlines: int
    fluidity_by_startup_delay: dict[float, Optional[AudioFluidityResult]]


def compute_interactivity_metrics(
    timing: RequestTiming,
    *,
    startup_delay_ms_values: Sequence[float],
    startup_buffer_ms_values: Sequence[float],
    min_reportable_stall_ms: float,
    fluidity_frame_ms: float,
    fluidity_startup_delay_ms: float,
    fluidity_attribution_mode: str,
) -> InteractivityMetrics:
    """Derive all playback policies from a single captured request timeline."""
    chunks = timing.audio_chunks
    first_audio_ms = chunks[0][0]
    last_audio_ms = chunks[-1][0]
    total_audio_ms = float(sum(chunk[1] for chunk in chunks))
    total_audio_bytes = float(sum(chunk[2] for chunk in chunks))

    playable_frame_arrivals = _playable_frame_arrival_ms(chunks, fluidity_frame_ms)
    first_playable_audio_ms = (
        playable_frame_arrivals[0] if playable_frame_arrivals else None
    )

    first_input_to_first_audio_ms = (
        first_audio_ms - timing.text_deltas[0][0] if timing.text_deltas else None
    )
    first_input_to_first_playable_audio_ms = (
        first_playable_audio_ms - timing.text_deltas[0][0]
        if timing.text_deltas and first_playable_audio_ms is not None
        else None
    )
    trigger_to_first_playable_audio_ms = (
        first_playable_audio_ms - timing.response_trigger_ms
        if timing.response_trigger_ms is not None
        and first_playable_audio_ms is not None
        else None
    )

    audio_before_commit_ratio: Optional[float] = None
    post_commit_audio_delivery_ms: Optional[float] = None
    duplex_overlap_observed = False
    duplex_overlap_ms = 0.0
    if timing.commit_ms is not None:
        bytes_before_commit = sum(
            n_bytes
            for receipt_ms, _, n_bytes in chunks
            if receipt_ms <= timing.commit_ms
        )
        audio_before_commit_ratio = (
            bytes_before_commit / total_audio_bytes if total_audio_bytes > 0 else None
        )
        post_commit_audio_delivery_ms = max(0.0, last_audio_ms - timing.commit_ms)
        if (
            first_playable_audio_ms is not None
            and first_playable_audio_ms < timing.commit_ms
        ):
            duplex_overlap_observed = True
            duplex_overlap_ms = max(
                0.0,
                min(last_audio_ms, timing.commit_ms) - first_playable_audio_ms,
            )

    fixed_delay_playback = {
        float(delay_ms): simulate_fixed_delay(
            chunks, float(delay_ms), min_reportable_stall_ms
        )
        for delay_ms in startup_delay_ms_values
    }
    # Chunks is non-empty, so fixed-delay simulations are always non-null.
    fixed_delay_playback = {
        delay_ms: result
        for delay_ms, result in fixed_delay_playback.items()
        if result is not None
    }

    buffer_target_playback = {
        float(target_ms): simulate_buffer_target(
            chunks,
            float(target_ms),
            min_reportable_stall_ms,
            audio_done_ms=timing.audio_done_ms,
            response_done_ms=timing.response_done_ms,
        )
        for target_ms in startup_buffer_ms_values
    }

    fluidity_delays = list(
        dict.fromkeys(
            [
                *(float(value) for value in startup_delay_ms_values),
                float(fluidity_startup_delay_ms),
            ]
        )
    )
    fluidity_by_startup_delay = {
        delay_ms: compute_audio_fluidity(
            chunks,
            frame_duration_ms=fluidity_frame_ms,
            startup_delay_ms=delay_ms,
        )
        for delay_ms in fluidity_delays
    }
    user_audio_fluidity = fluidity_by_startup_delay[float(fluidity_startup_delay_ms)]

    # A raw playback miss is always valid as a user-experience observation, but
    # it is not automatically attributable to the TTS service during duplex
    # input: the upstream text source may simply have supplied nothing to
    # synthesize. In conservative mode, publish a service score only when all
    # text was available before playback could begin. A controlled oversupply
    # trace may explicitly opt into service attribution.
    complete_input_before_playback = bool(
        timing.commit_ms is not None
        and first_playable_audio_ms is not None
        and timing.commit_ms <= first_playable_audio_ms
    )
    tts_service_fluidity_eligible = bool(
        user_audio_fluidity is not None
        and (
            complete_input_before_playback
            or fluidity_attribution_mode == "source_oversupplied"
        )
    )
    tts_service_fluidity = (
        user_audio_fluidity if tts_service_fluidity_eligible else None
    )
    unattributed_missed_deadlines = (
        0
        if tts_service_fluidity_eligible or user_audio_fluidity is None
        else user_audio_fluidity.missed_deadlines
    )

    streaming_rtf: Optional[float] = None
    if len(chunks) >= 2:
        delivered_after_first_ms = total_audio_ms - chunks[0][1]
        if delivered_after_first_ms > 0:
            streaming_rtf = (last_audio_ms - first_audio_ms) / delivered_after_first_ms

    done_after_last_audio_ms = (
        timing.response_done_ms - last_audio_ms
        if timing.response_done_ms is not None
        else None
    )

    return InteractivityMetrics(
        first_input_to_first_audio_ms=first_input_to_first_audio_ms,
        first_input_to_first_playable_audio_ms=(first_input_to_first_playable_audio_ms),
        trigger_to_first_playable_audio_ms=trigger_to_first_playable_audio_ms,
        request_start_to_first_audio_ms=first_audio_ms,
        request_start_to_first_playable_audio_ms=first_playable_audio_ms,
        audio_before_commit_ratio=audio_before_commit_ratio,
        duplex_overlap_observed=duplex_overlap_observed,
        duplex_overlap_ms=duplex_overlap_ms,
        post_commit_audio_delivery_ms=post_commit_audio_delivery_ms,
        required_startup_delay_ms=_required_startup_delay_ms(chunks),
        fixed_delay_playback=fixed_delay_playback,
        buffer_target_playback=buffer_target_playback,
        streaming_rtf=streaming_rtf,
        done_after_last_audio_ms=done_after_last_audio_ms,
        user_audio_fluidity=user_audio_fluidity,
        tts_service_fluidity=tts_service_fluidity,
        tts_service_fluidity_eligible=tts_service_fluidity_eligible,
        unattributed_missed_deadlines=unattributed_missed_deadlines,
        fluidity_by_startup_delay=fluidity_by_startup_delay,
    )
