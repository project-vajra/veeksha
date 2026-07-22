"""Audio measurement contract: PCM constants and the metric-key vocabulary.

Pure measurement facts shared by the audio (TTS / realtime) clients and the
audio evaluators. Transport-specific protocol handling lives in the clients.
"""

from __future__ import annotations

from enum import StrEnum

# 16-bit mono PCM at 24 kHz is the shared baseline across TTS dialects.
DEFAULT_AUDIO_SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
WAV_HEADER_BYTES = 44


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
    INPUT_CHARS = "input_chars"
    INPUT_TOKENS = "input_tokens"
    INPUT_TEXT = "input_text"
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
