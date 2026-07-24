"""Audio measurement contract: PCM constants and the metric-key vocabulary.

Pure measurement facts shared by the audio (TTS / realtime) clients and the
audio evaluators. Transport-specific protocol handling lives in the clients.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

# 16-bit mono PCM at 24 kHz is the shared baseline across TTS dialects.
DEFAULT_AUDIO_SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
WAV_HEADER_BYTES = 44


@dataclass(frozen=True)
class PCM16Audio:
    """Canonical mono PCM16 payload at the final client-consumer boundary."""

    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class AudioIntegrityMetrics:
    """Signal-integrity measurements over one canonical PCM sample array."""

    sample_count: int
    peak_abs_amplitude: float
    clipped_sample_fraction: float
    rms: float
    non_finite_sample_count: int


def decode_pcm16_audio(
    audio_data: bytes | bytearray | memoryview,
    *,
    raw_pcm: bool,
    sample_rate: int,
) -> PCM16Audio:
    """Decode raw or WAV-wrapped mono PCM16 into one canonical representation.

    Raw PCM and WAV samples are both interpreted as signed little-endian int16.
    Invalid or unsupported serialization is surfaced as ``ValueError`` instead
    of being treated as empty audio.
    """
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
                payload = wav_file.readframes(wav_file.getnframes())
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

    samples = np.frombuffer(payload, dtype="<i2")
    return PCM16Audio(samples=samples, sample_rate=decoded_sample_rate)


def measure_normalized_audio(samples: np.ndarray) -> AudioIntegrityMetrics:
    """Measure normalized audio, retaining diagnostics for non-finite values."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"Audio integrity measurement requires a 1-D array; got {values.ndim}-D"
        )

    sample_count = int(values.size)
    finite_mask = np.isfinite(values)
    non_finite_sample_count = int(sample_count - np.count_nonzero(finite_mask))
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        peak_abs_amplitude = 0.0
        clipped_sample_fraction = 0.0
        rms = 0.0
    else:
        abs_values = np.abs(finite_values)
        peak_abs_amplitude = float(np.max(abs_values))
        clipped_sample_fraction = (
            float(np.count_nonzero(abs_values >= 1.0) / sample_count)
            if sample_count > 0
            else 0.0
        )
        rms = float(np.sqrt(np.mean(np.square(finite_values), dtype=np.float64)))

    return AudioIntegrityMetrics(
        sample_count=sample_count,
        peak_abs_amplitude=peak_abs_amplitude,
        clipped_sample_fraction=clipped_sample_fraction,
        rms=rms,
        non_finite_sample_count=non_finite_sample_count,
    )


def measure_pcm16_audio(audio: PCM16Audio) -> AudioIntegrityMetrics:
    """Measure serialized PCM16 using exact int16 rail values for clipping."""
    samples = audio.samples
    sample_count = int(samples.size)
    if sample_count == 0:
        return AudioIntegrityMetrics(
            sample_count=0,
            peak_abs_amplitude=0.0,
            clipped_sample_fraction=0.0,
            rms=0.0,
            non_finite_sample_count=0,
        )

    clipped_sample_count = np.count_nonzero(
        (samples == np.iinfo(np.int16).min) | (samples == np.iinfo(np.int16).max)
    )
    min_sample = int(np.min(samples))
    max_sample = int(np.max(samples))
    peak_abs_amplitude = max(abs(min_sample), abs(max_sample)) / 32768.0
    float_samples = samples.astype(np.float64)
    rms = float(np.sqrt(np.dot(float_samples, float_samples) / sample_count) / 32768.0)

    return AudioIntegrityMetrics(
        sample_count=sample_count,
        peak_abs_amplitude=peak_abs_amplitude,
        clipped_sample_fraction=float(clipped_sample_count / sample_count),
        rms=rms,
        non_finite_sample_count=0,
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
    NON_FINITE_SAMPLE_COUNT = "non_finite_sample_count"
    AUDIO_SUSPECT = "audio_suspect"
    INPUT_CHARS = "input_chars"
    INPUT_TOKENS = "input_tokens"
    INPUT_TEXT = "input_text"
    SESSION_SIZE = "session_size"
    SESSION_DURATION = "session_duration"

    # True when the client deliberately closed the stream mid-utterance
    # (adversarial abort injection), rather than the server erroring. Aborted
    # requests are counted in their own bucket and excluded from continuity /
    # duration aggregates by the evaluator.
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
    RESPONSE_CREATED_OFFSET_MS = "response_created_offset_ms"  # nullable
    INPUT_COMMIT_OFFSET_MS = "input_commit_offset_ms"
    AUDIO_DONE_OFFSET_MS = "audio_done_offset_ms"  # nullable
    RESPONSE_DONE_OFFSET_MS = "response_done_offset_ms"  # nullable

    # Stable request-level interactivity keys emitted by the evaluator.
    FIRST_INPUT_TO_FIRST_AUDIO_MS = "first_input_to_first_audio_ms"
    REQUEST_START_TO_FIRST_AUDIO_MS = "request_start_to_first_audio_ms"
    AUDIO_BEFORE_COMMIT_RATIO = "audio_before_commit_ratio"
    POST_COMMIT_AUDIO_DELIVERY_MS = "post_commit_audio_delivery_ms"
    REQUIRED_STARTUP_DELAY_MS = "required_startup_delay_ms"
    ZERO_DELAY_STALL_COUNT = "zero_delay_stall_count"
    ZERO_DELAY_TOTAL_STALL_MS = "zero_delay_total_stall_ms"
    ZERO_DELAY_LONGEST_STALL_MS = "zero_delay_longest_stall_ms"
    ZERO_DELAY_STALL_FREE = "zero_delay_stall_free"

    # Diagnostic delivery/finalization metrics.
    STREAMING_RTF = "streaming_rtf"
    DONE_AFTER_LAST_AUDIO_MS = "done_after_last_audio_ms"
