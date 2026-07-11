"""TTS endpoint & provider routing for both transports.

One home for the per-dialect knowledge the TTS clients need to talk to a
server, kept out of the measurement contract in :mod:`veeksha.core.audio_contract`:

* Streaming **HTTP** TTS -- a small registry of provider entries (payload
  shape, streaming format, default key) plus request-URL construction.
* **Realtime** websocket TTS -- the OpenAI-Realtime event vocabulary, the
  single supported provider, and websocket-URL construction.

Both sections resolve a provider *name* -> its config *data*; keeping them side
by side is what makes that shared shape obvious.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional
from urllib.parse import quote, urljoin

OPENAI_V1_PREFIX = "v1"
AUDIO_SPEECH_PATH = "audio/speech"
OPENAI_AUDIO_SPEECH_PATH = f"{OPENAI_V1_PREFIX}/{AUDIO_SPEECH_PATH}"


# ---------------------------------------------------------------------------
# Streaming HTTP TTS providers
# ---------------------------------------------------------------------------


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


def get_tts_provider_entry(provider: str) -> TTSProviderEntry:
    """Return the provider entry for ``provider``; raise on an unknown name."""
    try:
        name = TTSProviderName(provider)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported TTS provider: {provider}. "
            f"Supported: {', '.join(SUPPORTED_TTS_PROVIDER_NAMES)}"
        ) from exc
    return TTS_PROVIDER_ENTRIES[name]


def _join_api_base_path(api_base: str, path: str) -> str:
    """Join ``api_base`` + ``path``, deduping a trailing ``/v1`` in the base
    against a leading ``v1/`` in the path (so ``.../v1`` + ``v1/audio/speech``
    yields ``.../v1/audio/speech``, not ``.../v1/v1/audio/speech``).
    """
    v1_prefix = f"{OPENAI_V1_PREFIX}/"
    if path.startswith(v1_prefix) and api_base.rstrip("/").endswith(
        f"/{OPENAI_V1_PREFIX}"
    ):
        path = path[len(v1_prefix) :]
    return urljoin(api_base.rstrip("/") + "/", path)


def build_tts_provider_url(api_base: str, provider_entry: TTSProviderEntry) -> str:
    return _join_api_base_path(api_base, provider_entry.path)


# ---------------------------------------------------------------------------
# Realtime (websocket) TTS
# ---------------------------------------------------------------------------


class RealtimeEventKind(StrEnum):
    SESSION_UPDATED = "session_updated"
    RESPONSE_CREATED = "response_created"
    AUDIO_DELTA = "audio_delta"
    AUDIO_DONE = "audio_done"
    RESPONSE_DONE = "response_done"
    ERROR = "error"
    OTHER = "other"


@dataclass(frozen=True)
class RealtimeTTSProvider:
    """Outbound frame types + inbound event vocabulary for a realtime provider.

    Defaults model the canonical GA OpenAI-Realtime protocol.
    """

    path: str = "v1/realtime"
    model_query_param: Optional[str] = "model"
    session_update_type: str = "session.update"
    # NON-SPEC EXTENSION: the OpenAI Realtime API has no incremental text buffer
    # (text is supplied whole via ``conversation.item.create``). These input
    # events, modelled on the spec's ``input_audio_buffer.*``, let the client
    # stream text deltas at an emulated LLM decode rate -- the core of this
    # benchmark. A conforming TTS server must accept them.
    input_append_type: str = "input_text_buffer.append"
    input_commit_type: str = "input_text_buffer.commit"
    response_create_type: Optional[str] = "response.create"
    session_updated_types: tuple[str, ...] = ("session.updated",)
    response_created_types: tuple[str, ...] = ("response.created",)
    # Audio output events were renamed in the GA API (``response.output_audio.*``);
    # accept both the GA and the older beta (``response.audio.*``) names.
    audio_delta_types: tuple[str, ...] = (
        "response.output_audio.delta",
        "response.audio.delta",
    )
    audio_done_types: tuple[str, ...] = (
        "response.output_audio.done",
        "response.audio.done",
    )
    response_done_types: tuple[str, ...] = ("response.done",)
    error_types: tuple[str, ...] = ("error", "response.error")
    audio_b64_field: str = "delta"
    force_raw_pcm: bool = True

    def raw_pcm(self, configured_raw_pcm: bool) -> bool:
        return self.force_raw_pcm or configured_raw_pcm


# Only one realtime provider is supported today (canonical OpenAI-Realtime). If a
# second server needs different event names, promote this to a name->provider
# registry mirroring TTS_PROVIDER_ENTRIES above.
OPENAI_REALTIME_PROVIDER = RealtimeTTSProvider()
SUPPORTED_REALTIME_TTS_PROVIDER_NAMES = ("openai_realtime",)


def get_realtime_tts_provider(provider: str) -> RealtimeTTSProvider:
    """Return the realtime provider for ``provider``; raise on an unknown name."""
    if provider not in SUPPORTED_REALTIME_TTS_PROVIDER_NAMES:
        raise ValueError(
            f"Unsupported realtime TTS provider: {provider}. "
            f"Supported: {', '.join(SUPPORTED_REALTIME_TTS_PROVIDER_NAMES)}"
        )
    return OPENAI_REALTIME_PROVIDER


def build_realtime_ws_url(
    api_base: str, provider: RealtimeTTSProvider, model: str
) -> str:
    """Build the websocket URL for a realtime TTS endpoint.

    Swaps the http(s) scheme for ws(s), joins ``provider.path`` via the shared
    ``_join_api_base_path`` helper, and appends ``?<param>=<model>`` when
    ``model_query_param`` is set and ``model`` is non-empty.
    """
    ws_base = api_base
    if ws_base.startswith("https://"):
        ws_base = "wss://" + ws_base[len("https://") :]
    elif ws_base.startswith("http://"):
        ws_base = "ws://" + ws_base[len("http://") :]

    url = _join_api_base_path(ws_base, provider.path)
    if provider.model_query_param and model:
        url = f"{url}?{provider.model_query_param}={quote(model, safe='')}"
    return url
