"""Shared audio metric, endpoint, and TTS provider contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urljoin

DEFAULT_AUDIO_SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
WAV_HEADER_BYTES = 44
OPENAI_V1_PREFIX = "v1"
AUDIO_SPEECH_PATH = "audio/speech"
OPENAI_AUDIO_SPEECH_PATH = f"{OPENAI_V1_PREFIX}/{AUDIO_SPEECH_PATH}"


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
    SESSION_SIZE = "session_size"
    SESSION_DURATION = "session_duration"


class TTSProviderName(StrEnum):
    VAJRA = "vajra"
    VAJRA_HIGGS = "vajra_higgs"
    VLLM_OMNI = "vllm_omni"
    SGLANG_OMNI = "sglang_omni"


class TTSStreamFormat(StrEnum):
    RAW_BYTES = "raw_bytes"
    SSE_AUDIO_JSON = "sse_audio_json"


class TTSPayloadFormat(StrEnum):
    OPENAI_SPEECH = "openai_speech"
    VAJRA_SYNTHESIZE = "vajra_synthesize"


@dataclass(frozen=True)
class TTSProviderEntry:
    name: TTSProviderName
    path: str = OPENAI_AUDIO_SPEECH_PATH
    payload_format: TTSPayloadFormat = TTSPayloadFormat.OPENAI_SPEECH
    default_api_key: str | None = None
    include_model: bool = False
    force_raw_pcm: bool = False
    stream_format: TTSStreamFormat = TTSStreamFormat.RAW_BYTES

    def raw_pcm(self, configured_raw_pcm: bool) -> bool:
        return self.force_raw_pcm or configured_raw_pcm

    def response_format(self, configured_raw_pcm: bool) -> str:
        return "pcm" if self.raw_pcm(configured_raw_pcm) else "wav"


TTS_PROVIDER_ENTRIES: Mapping[TTSProviderName, TTSProviderEntry] = {
    TTSProviderName.VAJRA: TTSProviderEntry(TTSProviderName.VAJRA),
    TTSProviderName.VAJRA_HIGGS: TTSProviderEntry(
        TTSProviderName.VAJRA_HIGGS,
        path="synthesize/stream",
        payload_format=TTSPayloadFormat.VAJRA_SYNTHESIZE,
    ),
    TTSProviderName.VLLM_OMNI: TTSProviderEntry(
        TTSProviderName.VLLM_OMNI,
        default_api_key="EMPTY",
        include_model=True,
        force_raw_pcm=True,
    ),
    TTSProviderName.SGLANG_OMNI: TTSProviderEntry(
        TTSProviderName.SGLANG_OMNI,
        include_model=True,
        force_raw_pcm=True,
        stream_format=TTSStreamFormat.SSE_AUDIO_JSON,
    ),
}

SUPPORTED_TTS_PROVIDER_NAMES = tuple(
    provider.value for provider in TTS_PROVIDER_ENTRIES
)


def normalize_tts_provider(provider: str) -> TTSProviderName:
    try:
        return TTSProviderName(provider)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported TTS provider: {provider}. "
            f"Supported: {', '.join(SUPPORTED_TTS_PROVIDER_NAMES)}"
        ) from exc


def get_tts_provider_entry(provider: str) -> TTSProviderEntry:
    return TTS_PROVIDER_ENTRIES[normalize_tts_provider(provider)]


def build_tts_provider_url(api_base: str, provider_entry: TTSProviderEntry) -> str:
    endpoint = provider_entry.path
    if (
        endpoint == OPENAI_AUDIO_SPEECH_PATH
        and api_base.rstrip("/").endswith(f"/{OPENAI_V1_PREFIX}")
    ):
        endpoint = AUDIO_SPEECH_PATH
    return urljoin(api_base.rstrip("/") + "/", endpoint)


def build_audio_speech_url(api_base: str) -> str:
    return build_tts_provider_url(api_base, TTS_PROVIDER_ENTRIES[TTSProviderName.VAJRA])
