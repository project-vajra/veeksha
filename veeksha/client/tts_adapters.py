"""Per-provider TTS dialect adapters, shared by both TTS clients.

Two transports, one concept: an *adapter* wraps a provider entry and knows how
to build outbound requests/frames and decode inbound audio for one server
dialect, keeping that dialect-specific knowledge out of the client bodies.

- :class:`TTSProviderAdapter` -- streaming HTTP (used by
  :class:`~veeksha.client.tts.TTSClient`).
- :class:`RealtimeTTSAdapter` -- OpenAI-Realtime-style websocket (used by
  :class:`~veeksha.client.realtime_tts.RealtimeTTSClient`).
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import httpx

from veeksha.core.tts_providers import (
    RealtimeEventKind,
    RealtimeTTSProvider,
    TTSPayloadFormat,
    TTSProviderEntry,
    TTSStreamFormat,
    build_realtime_ws_url,
    build_tts_provider_url,
    get_realtime_tts_provider,
    get_tts_provider_entry,
)

if TYPE_CHECKING:
    from veeksha.config.client import RealtimeTTSClientConfig, TTSClientConfig


# ---------------------------------------------------------------------------
# Streaming HTTP TTS adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TTSProviderRequest:
    url: str
    headers: dict[str, str]
    payload: dict


class TTSProviderAdapter:
    """Request adapter for a configured streaming TTS provider."""

    def __init__(
        self, config: TTSClientConfig, provider_entry: TTSProviderEntry
    ) -> None:
        self.config = config
        self.provider_entry = provider_entry

    @property
    def raw_pcm(self) -> bool:
        """Whether the provider streams raw PCM bytes."""
        return self.provider_entry.raw_pcm(self.config.raw_pcm)

    def build_request(self, text: str) -> TTSProviderRequest:
        """Build a streaming HTTP request for a text input."""
        if self.provider_entry.payload_format is TTSPayloadFormat.VAJRA_SYNTHESIZE:
            payload: dict = {"text": text}
            if self.config.voice_id:
                payload["speaker"] = self.config.voice_id
        else:
            payload = {
                "input": text,
                "response_format": self.provider_entry.response_format(
                    self.config.raw_pcm
                ),
                "stream": True,
            }
            if self.provider_entry.include_model:
                payload["model"] = self.config.model
            if self.config.voice_id:
                payload["voice"] = self.config.voice_id

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        return TTSProviderRequest(
            url=build_tts_provider_url(str(self.config.api_base), self.provider_entry),
            headers=headers,
            payload=payload,
        )

    async def iter_audio_chunks(
        self, response: httpx.Response, chunk_size: int
    ) -> AsyncIterator[bytes]:
        """Yield decoded audio chunks from a streaming response."""
        if self.provider_entry.stream_format is TTSStreamFormat.SSE_AUDIO_JSON:
            async for chunk in self._iter_sse_audio_chunks(response):
                yield chunk
            return

        async for chunk in response.aiter_bytes(chunk_size=chunk_size):
            if chunk:
                yield chunk

    async def _iter_sse_audio_chunks(
        self, response: httpx.Response
    ) -> AsyncIterator[bytes]:
        async for line in response.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            audio = event.get("audio")
            if not isinstance(audio, dict):
                continue
            encoded = audio.get("data")
            if not encoded:
                continue
            yield base64.b64decode(encoded)


def _build_provider_adapter(config: TTSClientConfig) -> TTSProviderAdapter:
    return TTSProviderAdapter(config, get_tts_provider_entry(config.provider))


# ---------------------------------------------------------------------------
# Realtime websocket TTS adapter
# ---------------------------------------------------------------------------


class RealtimeTTSAdapter:
    """Builds outbound frames and classifies inbound events for a dialect."""

    def __init__(
        self,
        config: "RealtimeTTSClientConfig",
        entry: RealtimeTTSProvider,
        api_key: Optional[str] = None,
    ) -> None:
        self.config = config
        self.entry = entry
        # ``api_key`` lets the client pass its env-resolved key; fall back to the
        # config value for direct (test) construction.
        self._api_key = api_key if api_key is not None else config.api_key

        # Precompute the inbound event-type -> kind lookup from the entry tuples.
        kinds_by_types = [
            (RealtimeEventKind.SESSION_UPDATED, entry.session_updated_types),
            (RealtimeEventKind.RESPONSE_CREATED, entry.response_created_types),
            (RealtimeEventKind.AUDIO_DELTA, entry.audio_delta_types),
            (RealtimeEventKind.AUDIO_DONE, entry.audio_done_types),
            (RealtimeEventKind.RESPONSE_DONE, entry.response_done_types),
            (RealtimeEventKind.ERROR, entry.error_types),
        ]
        self._event_kind: dict[str, RealtimeEventKind] = {
            event_type: kind
            for kind, event_types in kinds_by_types
            for event_type in event_types
        }

    @property
    def raw_pcm(self) -> bool:
        """Whether the provider streams raw PCM (force wins; effectively True)."""
        return self.entry.raw_pcm(self.config.raw_pcm)

    def build_ws_url(self, api_base: str) -> str:
        return build_realtime_ws_url(api_base, self.entry, self.config.model)

    def headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def session_update_json(self) -> str:
        # GA session shape: output audio format is a nested object under
        # ``audio.output``; the model is carried on the URL (not the session),
        # so it is intentionally omitted here.
        output: dict = {
            "format": {"type": "audio/pcm", "rate": self.config.sample_rate}
        }
        if self.config.voice_id:
            output["voice"] = self.config.voice_id
        session = {"type": "realtime", "audio": {"output": output}}
        return json.dumps({"type": self.entry.session_update_type, "session": session})

    def input_append_json(self, text: str) -> str:
        return json.dumps({"type": self.entry.input_append_type, "text": text})

    def input_commit_json(self) -> str:
        return json.dumps({"type": self.entry.input_commit_type})

    def response_create_json(self) -> Optional[str]:
        """Frame for an explicit response trigger, or ``None`` if commit implies it."""
        if self.entry.response_create_type is None:
            return None
        return json.dumps({"type": self.entry.response_create_type})

    def classify(self, event_type: str) -> RealtimeEventKind:
        return self._event_kind.get(event_type, RealtimeEventKind.OTHER)

    def extract_audio(self, event: dict) -> bytes:
        encoded = event.get(self.entry.audio_b64_field)
        if not encoded:
            return b""
        return base64.b64decode(encoded)


def _build_realtime_adapter(
    config: "RealtimeTTSClientConfig", api_key: Optional[str] = None
) -> RealtimeTTSAdapter:
    return RealtimeTTSAdapter(
        config, get_realtime_tts_provider(config.provider), api_key=api_key
    )
