"""Audio measurement contract: PCM constants and the metric-key vocabulary.

Pure measurement facts shared by the audio (TTS / realtime) clients and the
audio evaluators. Transport-specific protocol handling lives in the clients.
"""

from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

# 16-bit mono PCM at 24 kHz is the shared baseline across TTS dialects.
DEFAULT_AUDIO_SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
_MEASUREMENT_BLOCK_SAMPLES = 1 << 20


@dataclass(frozen=True)
class PCM16Audio:
    """Canonical mono PCM16 payload at the final consumer boundary."""

    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class AudioIntegrityMetrics:
    """Signal-integrity measurements over canonical PCM16 samples."""

    sample_count: int
    peak_abs_amplitude: float
    clipped_sample_fraction: float
    rms: float


def decode_pcm16_audio(
    audio_data: bytes | bytearray | memoryview,
    *,
    raw_pcm: bool,
    sample_rate: int,
) -> PCM16Audio:
    """Decode raw or WAV-wrapped mono PCM16 into one representation."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0; got {sample_rate}")
    if not isinstance(audio_data, (bytes, bytearray, memoryview)):
        raise TypeError(
            "audio_data must be bytes, bytearray, or memoryview; "
            f"got {type(audio_data).__name__}"
        )

    payload = bytes(audio_data)
    decoded_sample_rate = sample_rate
    if not raw_pcm:
        try:
            with wave.open(io.BytesIO(payload), "rb") as wav_file:
                if wav_file.getnchannels() != 1:
                    raise ValueError(
                        "Audio integrity measurement requires mono WAV input; "
                        f"got {wav_file.getnchannels()} channels"
                    )
                if wav_file.getsampwidth() != BYTES_PER_SAMPLE:
                    raise ValueError(
                        "Audio integrity measurement requires 16-bit WAV input; "
                        f"got {wav_file.getsampwidth() * 8}-bit samples"
                    )
                if wav_file.getcomptype() != "NONE":
                    raise ValueError(
                        "Audio integrity measurement requires uncompressed PCM WAV "
                        f"input; got compression {wav_file.getcomptype()!r}"
                    )
                decoded_sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                payload = wav_file.readframes(frame_count)
                expected_payload_bytes = frame_count * BYTES_PER_SAMPLE
                if len(payload) != expected_payload_bytes:
                    raise ValueError(
                        "WAV payload is shorter than its declared frame count; "
                        f"expected {expected_payload_bytes} PCM bytes, "
                        f"got {len(payload)}"
                    )
        except (EOFError, wave.Error) as error:
            raise ValueError(f"Invalid WAV payload: {error}") from error

    if decoded_sample_rate <= 0:
        raise ValueError(
            f"Decoded PCM16 sample rate must be > 0; got {decoded_sample_rate}"
        )
    if len(payload) % BYTES_PER_SAMPLE != 0:
        raise ValueError(
            "PCM16 payload length must be divisible by 2 bytes; "
            f"got {len(payload)} bytes"
        )

    return PCM16Audio(
        samples=np.frombuffer(payload, dtype="<i2"),
        sample_rate=decoded_sample_rate,
    )


def measure_pcm16_audio(audio: PCM16Audio) -> AudioIntegrityMetrics:
    """Measure PCM16 amplitude, clipping, and RMS with bounded scratch memory."""
    samples = audio.samples
    sample_count = int(samples.size)
    if sample_count == 0:
        return AudioIntegrityMetrics(
            sample_count=0,
            peak_abs_amplitude=0.0,
            clipped_sample_fraction=0.0,
            rms=0.0,
        )

    int16_info = np.iinfo(np.int16)
    peak_sample = 0
    clipped_sample_count = 0
    sum_squares = 0.0
    for start in range(0, sample_count, _MEASUREMENT_BLOCK_SAMPLES):
        block = samples[start : start + _MEASUREMENT_BLOCK_SAMPLES]
        min_sample = int(np.min(block))
        max_sample = int(np.max(block))
        peak_sample = max(peak_sample, abs(min_sample), abs(max_sample))
        clipped_sample_count += int(
            np.count_nonzero((block == int16_info.min) | (block == int16_info.max))
        )
        float_block = block.astype(np.float64)
        sum_squares += float(np.dot(float_block, float_block))

    return AudioIntegrityMetrics(
        sample_count=sample_count,
        peak_abs_amplitude=peak_sample / 32768.0,
        clipped_sample_fraction=clipped_sample_count / sample_count,
        rms=math.sqrt(sum_squares / sample_count) / 32768.0,
    )


def pcm_bytes_to_duration_ms(
    n_bytes: float, sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
) -> float:
    """Duration in ms of ``n_bytes`` of 16-bit mono PCM at ``sample_rate``."""
    return n_bytes / (sample_rate * BYTES_PER_SAMPLE) * 1000.0


class AudioMetricKey(StrEnum):
    TTFC = "ttfc"
    END_TO_END_LATENCY = "end_to_end_latency"
    GENERATED_AUDIO_DURATION = "generated_audio_duration"
    RTF = "rtf"
    CHUNK_COUNT = "chunk_count"
    RAW_PCM = "raw_pcm"
    SAMPLE_RATE = "sample_rate"
    PCM_BYTE_COUNT = "pcm_byte_count"
    PCM_SAMPLE_COUNT = "pcm_sample_count"
    PEAK_ABS_AMPLITUDE = "peak_abs_amplitude"
    CLIPPED_SAMPLE_FRACTION = "clipped_sample_fraction"
    RMS = "rms"
    AUDIO_SUSPECT = "audio_suspect"
    INPUT_CHARS = "input_chars"
    INPUT_TOKENS = "input_tokens"
    INPUT_TEXT = "input_text"
    TEXT_PACING_UNIT = "text_pacing_unit"
    TEXT_PACING_RATE = "text_pacing_rate"
    PROVIDER = "provider"
    PROVIDER_MODEL = "provider_model"
    PROVIDER_PROTOCOL = "provider_protocol"
    SESSION_SIZE = "session_size"
    SESSION_DURATION = "session_duration"
    ABORTED = "aborted"

    # ----- Realtime input-streaming interactivity keys -----
    # Time convention for all realtime event-time values below: every *_offset_ms
    # / timestamp value is a float millisecond offset relative to request start
    # (the client's WS-connect initiation), measured with time.monotonic().
    #
    # Raw-contract keys are emitted by the websocket client:
    TEXT_DELTA_TIMESTAMPS = "text_delta_timestamps"  # list[[offset_ms, n_chars]]
    AUDIO_CHUNK_TIMESTAMPS = (
        "audio_chunk_timestamps"  # list[[offset_ms, n_bytes_decoded_pcm]]
    )
    WS_CONNECT_LATENCY_MS = "ws_connect_latency_ms"
    SESSION_READY_OFFSET_MS = "session_ready_offset_ms"  # nullable
    RESPONSE_TRIGGER_OFFSET_MS = "response_trigger_offset_ms"
    RESPONSE_CREATED_OFFSET_MS = "response_created_offset_ms"  # nullable
    INPUT_COMMIT_OFFSET_MS = "input_commit_offset_ms"
    AUDIO_DONE_OFFSET_MS = "audio_done_offset_ms"  # nullable
    RESPONSE_DONE_OFFSET_MS = "response_done_offset_ms"  # nullable

    # Stable request-level interactivity keys emitted by the evaluator.
    FIRST_INPUT_TO_FIRST_AUDIO_MS = "first_input_to_first_audio_ms"
    FIRST_INPUT_TO_FIRST_PLAYABLE_AUDIO_MS = "first_input_to_first_playable_audio_ms"
    TRIGGER_TO_FIRST_PLAYABLE_AUDIO_MS = "trigger_to_first_playable_audio_ms"
    REQUEST_START_TO_FIRST_AUDIO_MS = "request_start_to_first_audio_ms"
    REQUEST_START_TO_FIRST_PLAYABLE_AUDIO_MS = (
        "request_start_to_first_playable_audio_ms"
    )
    AUDIO_BEFORE_COMMIT_RATIO = "audio_before_commit_ratio"
    DUPLEX_OVERLAP_OBSERVED = "duplex_overlap_observed"
    DUPLEX_OVERLAP_MS = "duplex_overlap_ms"
    POST_COMMIT_AUDIO_DELIVERY_MS = "post_commit_audio_delivery_ms"
    REQUIRED_STARTUP_DELAY_MS = "required_startup_delay_ms"
    ZERO_DELAY_STALL_COUNT = "zero_delay_stall_count"
    ZERO_DELAY_TOTAL_STALL_MS = "zero_delay_total_stall_ms"
    ZERO_DELAY_LONGEST_STALL_MS = "zero_delay_longest_stall_ms"
    ZERO_DELAY_STALL_FREE = "zero_delay_stall_free"

    # Diagnostic delivery/finalization metrics.
    STREAMING_RTF = "streaming_rtf"
    DONE_AFTER_LAST_AUDIO_MS = "done_after_last_audio_ms"

    # Etalon-inspired playable-frame deadline metrics. The untagged user score
    # uses AudioChannelPerformanceConfig.fluidity_startup_delay_ms.
    USER_AUDIO_FLUIDITY_INDEX = "user_audio_fluidity_index"
    TTS_SERVICE_FLUIDITY_INDEX = "tts_service_fluidity_index"
    TTS_SERVICE_FLUIDITY_ELIGIBLE = "tts_service_fluidity_eligible"
    UNATTRIBUTED_MISSED_DEADLINES = "unattributed_missed_deadlines"
    AUDIO_FLUIDITY_TOTAL_DEADLINES = "audio_fluidity_total_deadlines"
    AUDIO_FLUIDITY_MISSED_DEADLINES = "audio_fluidity_missed_deadlines"
    AUDIO_PLAYABLE_FRAME_COUNT = "audio_playable_frame_count"
    AUDIO_FLUIDITY_FRAME_MS = "audio_fluidity_frame_ms"
    AUDIO_FLUIDITY_STARTUP_DELAY_MS = "audio_fluidity_startup_delay_ms"
