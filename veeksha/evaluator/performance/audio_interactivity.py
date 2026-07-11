"""Pure computation for realtime TTS interactivity metrics.

The websocket client records one monotonic timeline per request. This module
replays that timeline locally under multiple playback policies; no additional
server requests are made for a policy sweep.
"""

from __future__ import annotations

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
    request_start_to_first_audio_ms: float
    audio_before_commit_ratio: Optional[float]
    post_commit_audio_delivery_ms: Optional[float]
    required_startup_delay_ms: float
    fixed_delay_playback: dict[float, PlaybackSimResult]
    buffer_target_playback: dict[float, Optional[PlaybackSimResult]]
    streaming_rtf: Optional[float]
    done_after_last_audio_ms: Optional[float]


def compute_interactivity_metrics(
    timing: RequestTiming,
    *,
    startup_delay_ms_values: Sequence[float],
    startup_buffer_ms_values: Sequence[float],
    min_reportable_stall_ms: float,
) -> InteractivityMetrics:
    """Derive all playback policies from a single captured request timeline."""
    chunks = timing.audio_chunks
    first_audio_ms = chunks[0][0]
    last_audio_ms = chunks[-1][0]
    total_audio_ms = float(sum(chunk[1] for chunk in chunks))
    total_audio_bytes = float(sum(chunk[2] for chunk in chunks))

    first_input_to_first_audio_ms = (
        first_audio_ms - timing.text_deltas[0][0] if timing.text_deltas else None
    )

    audio_before_commit_ratio: Optional[float] = None
    post_commit_audio_delivery_ms: Optional[float] = None
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
        request_start_to_first_audio_ms=first_audio_ms,
        audio_before_commit_ratio=audio_before_commit_ratio,
        post_commit_audio_delivery_ms=post_commit_audio_delivery_ms,
        required_startup_delay_ms=_required_startup_delay_ms(chunks),
        fixed_delay_playback=fixed_delay_playback,
        buffer_target_playback=buffer_target_playback,
        streaming_rtf=streaming_rtf,
        done_after_last_audio_ms=done_after_last_audio_ms,
    )
