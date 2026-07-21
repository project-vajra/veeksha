from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
import types
from typing import Any

import pytest

if (
    "transformers" not in sys.modules
    and importlib.util.find_spec("transformers") is None
):
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = object  # type: ignore[attr-defined]
    sys.modules["transformers"] = transformers_stub

from veeksha.client.native_streaming_tts import (
    _ERROR_PRIORITY,
    DeepgramAuraStreamingProtocol,
    DeepgramAuraStreamingTTSClient,
    DeepgramFluxStreamingProtocol,
    DeepgramFluxStreamingTTSClient,
    ElevenLabsStreamingProtocol,
    ElevenLabsStreamingTTSClient,
    NativeStreamingTTSError,
    _map_error,
)
from veeksha.client.utils import flatten_ws_exception
from veeksha.config.client import (
    DeepgramAuraStreamingTTSClientConfig,
    DeepgramFluxStreamingTTSClientConfig,
    ElevenLabsStreamingTTSClientConfig,
    TextPacingConfig,
)
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.types import ChannelModality


class _FakeWebSocket:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.sent: list[str | bytes] = []
        self.events: asyncio.Queue[str | bytes] = asyncio.Queue()
        self.speech_started = False
        if provider == "deepgram":
            self.events.put_nowait(json.dumps({"type": "Connected"}))

    async def send(self, raw: str | bytes) -> None:
        self.sent.append(raw)
        event = json.loads(raw)
        if self.provider == "elevenlabs":
            if event.get("text") and event.get("text") != " ":
                audio = base64.b64encode(bytes(960)).decode("ascii")
                await self.events.put(json.dumps({"audio": audio, "isFinal": False}))
            elif event.get("text") == "":
                await self.events.put(json.dumps({"isFinal": True}))
        elif self.provider == "deepgram_aura" and event.get("type") == "Flush":
            await self.events.put(bytes(960))
            await self.events.put(json.dumps({"type": "Flushed", "sequence_id": 0}))
        elif event.get("type") == "Speak" and self.provider == "deepgram":
            if not self.speech_started:
                await self.events.put(json.dumps({"type": "SpeechStarted"}))
                self.speech_started = True
            await self.events.put(bytes(960))
        elif event.get("type") == "Flush":
            await self.events.put(json.dumps({"type": "Flushed"}))
            await self.events.put(
                json.dumps(
                    {
                        "type": "SpeechMetadata",
                        "audio_duration_ms": 60,
                        "input_character_count": 13,
                    }
                )
            )

    async def recv(self) -> str | bytes:
        return await self.events.get()


class _FakeConnection:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> _FakeWebSocket:
        return self.websocket

    async def __aexit__(self, *args: object) -> None:
        return None


def _request() -> Request:
    return Request(
        id=11,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(input_text="one two three")
        },
    )


def _pacing() -> TextPacingConfig:
    return TextPacingConfig(tokens_per_second=1000, tokens_per_delta=1)


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["elevenlabs", "deepgram"])
def test_native_clients_stream_text_and_audio_concurrently(provider: str) -> None:
    if provider == "elevenlabs":
        config = ElevenLabsStreamingTTSClientConfig(
            api_base="https://api.elevenlabs.io",
            api_key="test-key",
            model="eleven_flash_v2_5",
            voice_id="test-voice",
            pacing=_pacing(),
        )
        client: Any = ElevenLabsStreamingTTSClient(config)
    else:
        config = DeepgramFluxStreamingTTSClientConfig(
            api_base="https://api.deepgram.com",
            api_key="test-key",
            model="flux-alexis-en",
            pacing=_pacing(),
        )
        client = DeepgramFluxStreamingTTSClient(config)

    websocket = _FakeWebSocket(provider)
    client._connect = lambda: _FakeConnection(websocket)  # type: ignore[method-assign]
    result = asyncio.run(client.send_request(_request(), session_id=1))

    assert result.success
    metrics = result.channels[ChannelModality.AUDIO].metrics
    assert metrics[AudioMetricKey.PROVIDER.value] == provider
    assert metrics[AudioMetricKey.PROVIDER_MODEL.value] == config.model
    assert len(metrics[AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value]) == 3
    assert metrics[AudioMetricKey.CHUNK_COUNT.value] == 3
    assert metrics[AudioMetricKey.RESPONSE_TRIGGER_OFFSET_MS.value] == pytest.approx(
        metrics[AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value][0][0], abs=0.001
    )
    assert (
        metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value][0][0]
        < metrics[AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value]
    )


@pytest.mark.unit
def test_native_protocol_urls_use_raw_pcm_and_provider_auth() -> None:
    eleven_config = ElevenLabsStreamingTTSClientConfig(
        api_base="https://api.elevenlabs.io",
        api_key="eleven-key",
        model="eleven_flash_v2_5",
        voice_id="voice/id",
    )
    eleven = ElevenLabsStreamingProtocol(eleven_config, "eleven-key")
    assert "/voice%2Fid/stream-input" in eleven.build_ws_url(
        str(eleven_config.api_base)
    )
    assert "output_format=pcm_24000" in eleven.build_ws_url(str(eleven_config.api_base))
    assert eleven.headers() == {"xi-api-key": "eleven-key"}

    deepgram_config = DeepgramFluxStreamingTTSClientConfig(
        api_base="https://api.deepgram.com",
        api_key="deepgram-key",
        model="flux-alexis-en",
    )
    deepgram = DeepgramFluxStreamingProtocol(deepgram_config, "deepgram-key")
    assert "/v2/speak?" in deepgram.build_ws_url(str(deepgram_config.api_base))
    assert "encoding=linear16" in deepgram.build_ws_url(str(deepgram_config.api_base))
    assert "speed=" not in deepgram.build_ws_url(str(deepgram_config.api_base))
    assert deepgram.headers() == {"Authorization": "Token deepgram-key"}

    aura_config = DeepgramAuraStreamingTTSClientConfig(
        api_base="https://api.deepgram.com",
        api_key="deepgram-key",
        model="aura-2-thalia-en",
    )
    aura = DeepgramAuraStreamingProtocol(aura_config, "deepgram-key")
    assert "/v1/speak?" in aura.build_ws_url(str(aura_config.api_base))
    assert "encoding=linear16" in aura.build_ws_url(str(aura_config.api_base))
    assert "speed=1.0" in aura.build_ws_url(str(aura_config.api_base))
    assert aura.headers() == {"Authorization": "Token deepgram-key"}


@pytest.mark.unit
def test_deepgram_aura_adapter_handles_audio_after_flush() -> None:
    config = DeepgramAuraStreamingTTSClientConfig(
        api_base="https://api.deepgram.com",
        api_key="test-key",
        model="aura-2-thalia-en",
        pacing=_pacing(),
    )
    client = DeepgramAuraStreamingTTSClient(config)
    websocket = _FakeWebSocket("deepgram_aura")
    client._connect = lambda: _FakeConnection(websocket)  # type: ignore[method-assign]

    result = asyncio.run(client.send_request(_request(), session_id=1))

    assert result.success
    metrics = result.channels[ChannelModality.AUDIO].metrics
    assert metrics[AudioMetricKey.PROVIDER_PROTOCOL.value] == "v1_aura_speak"
    first_audio_ms = metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value][0][0]
    commit_ms = metrics[AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value]
    trigger_ms = metrics[AudioMetricKey.RESPONSE_TRIGGER_OFFSET_MS.value]
    assert trigger_ms < commit_ms
    assert first_audio_ms >= commit_ms
    assert [json.loads(raw)["type"] for raw in websocket.sent] == [
        "Speak",
        "Speak",
        "Speak",
        "Flush",
    ]


@pytest.mark.unit
def test_native_client_requires_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    config = DeepgramFluxStreamingTTSClientConfig(
        api_base="https://api.deepgram.com",
        model="flux-alexis-en",
    )
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        DeepgramFluxStreamingTTSClient(config)


@pytest.mark.unit
def test_native_streaming_error_priority_preserves_provider_error() -> None:
    provider_error = NativeStreamingTTSError("provider failed")
    grouped = ExceptionGroup(
        "task failures",
        [OSError("socket closed"), ExceptionGroup("provider", [provider_error])],
    )

    flattened = flatten_ws_exception(grouped, _ERROR_PRIORITY)

    assert flattened is provider_error
    assert _map_error(flattened) == (500, "provider failed")
