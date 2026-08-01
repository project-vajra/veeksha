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

from veeksha.client.streaming_tts import (
    _ERROR_PRIORITY,
    CartesiaStreamingProtocol,
    DeepgramAuraStreamingProtocol,
    DeepgramFluxStreamingProtocol,
    ElevenLabsStreamingProtocol,
    RawAudioEvent,
    StreamingTTSClient,
    StreamingTTSError,
    VajraStreamingProtocol,
    _map_error,
)
from veeksha.client.utils import TextDeltaPacer, flatten_ws_exception, segment_text
from veeksha.config.client import StreamingTTSClientConfig, TextPacingConfig
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
        elif self.provider == "deepgram_aura" and event.get("type") == "Speak":
            await self.events.put(bytes(960))
        elif self.provider == "cartesia":
            if event.get("continue") is True:
                audio = base64.b64encode(bytes(960)).decode("ascii")
                await self.events.put(
                    json.dumps({"type": "chunk", "data": audio, "done": False})
                )
            elif event.get("continue") is False:
                await self.events.put(json.dumps({"type": "done", "done": True}))
        elif self.provider == "deepgram_aura" and event.get("type") == "Flush":
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


class _FakePingWebSocket:
    def __init__(self, rtt_s: float) -> None:
        self.rtt_s = rtt_s

    async def ping(self) -> asyncio.Future[float]:
        pong_received = asyncio.get_running_loop().create_future()
        pong_received.set_result(self.rtt_s)
        return pong_received


class _SilentElevenLabsWebSocket(_FakeWebSocket):
    def __init__(self) -> None:
        super().__init__("elevenlabs")

    async def send(self, raw: str | bytes) -> None:
        self.sent.append(raw)
        event = json.loads(raw)
        if event.get("text") == "":
            await self.events.put(json.dumps({"isFinal": True}))


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
def test_text_pacing_uses_whitespace_words_at_a_continuous_rate() -> None:
    pacing = TextPacingConfig(tokens_per_second=2, tokens_per_delta=1)
    segments = segment_text("one two three", pacing.tokens_per_delta)
    pacer = TextDeltaPacer(pacing, seed=42)

    assert [segment.text for segment in segments] == ["one ", "two ", "three"]
    assert [pacer.next_gap() for _ in segments] == [0.5, 0.5, 0.5]


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["elevenlabs", "deepgram", "cartesia"])
def test_streaming_providers_share_text_audio_lifecycle(provider: str) -> None:
    if provider == "elevenlabs":
        config = StreamingTTSClientConfig(
            provider="elevenlabs",
            api_base="https://api.elevenlabs.io",
            api_key="test-key",
            model="eleven_flash_v2_5",
            voice_id="test-voice",
            pacing=_pacing(),
        )
        client: Any = StreamingTTSClient(config)
    elif provider == "deepgram":
        config = StreamingTTSClientConfig(
            provider="deepgram_flux",
            api_base="https://api.deepgram.com",
            api_key="test-key",
            model="flux-alexis-en",
            pacing=_pacing(),
        )
        client = StreamingTTSClient(config)
    else:
        config = StreamingTTSClientConfig(
            provider="cartesia",
            api_base="https://api.cartesia.ai",
            api_key="test-key",
            model="sonic-3.5",
            voice_id="test-voice",
            language="en",
            pacing=_pacing(),
        )
        client = StreamingTTSClient(config)

    websocket = _FakeWebSocket(provider)
    client._connect = lambda *_args, **_kwargs: _FakeConnection(websocket)  # type: ignore[method-assign]
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
    assert metrics[AudioMetricKey.TTFC.value] == pytest.approx(
        metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value][0][0]
        - metrics[AudioMetricKey.RESPONSE_TRIGGER_OFFSET_MS.value],
        abs=1.0,
    )
    assert metrics[AudioMetricKey.TEXT_PACING_UNIT.value] == "whitespace_word"
    assert metrics[AudioMetricKey.TEXT_PACING_RATE.value] == 1000
    assert (
        metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value][0][0]
        < metrics[AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value]
    )


@pytest.mark.unit
def test_websocket_rtt_probe_uses_independent_connections() -> None:
    config = StreamingTTSClientConfig(
        provider="deepgram_aura",
        api_base="https://api.deepgram.com",
        api_key="test-key",
        model="aura-2-thalia-en",
    )
    client = StreamingTTSClient(config)
    pending_rtt_s = iter((0.010, 0.020, 0.030))
    opened_websockets: list[_FakePingWebSocket] = []

    def connect_factory() -> _FakeConnection:
        websocket = _FakePingWebSocket(next(pending_rtt_s))
        opened_websockets.append(websocket)
        return _FakeConnection(websocket)  # type: ignore[arg-type]

    client._connect = connect_factory  # type: ignore[method-assign]

    samples = asyncio.run(client.measure_websocket_rtt_ms(samples=3))

    assert samples == pytest.approx([10.0, 20.0, 30.0])
    assert len(opened_websockets) == 3


@pytest.mark.unit
def test_websocket_rtt_probe_rejects_nonpositive_sample_count() -> None:
    config = StreamingTTSClientConfig(
        provider="deepgram_aura",
        api_base="https://api.deepgram.com",
        api_key="test-key",
        model="aura-2-thalia-en",
    )
    client = StreamingTTSClient(config)

    with pytest.raises(ValueError, match="samples must be >= 1"):
        asyncio.run(client.measure_websocket_rtt_ms(samples=0))


@pytest.mark.unit
def test_streaming_protocol_urls_use_raw_pcm_and_provider_auth() -> None:
    eleven_config = StreamingTTSClientConfig(
        provider="elevenlabs",
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

    deepgram_config = StreamingTTSClientConfig(
        provider="deepgram_flux",
        api_base="https://api.deepgram.com",
        api_key="deepgram-key",
        model="flux-alexis-en",
    )
    deepgram = DeepgramFluxStreamingProtocol(deepgram_config, "deepgram-key")
    assert "/v2/speak?" in deepgram.build_ws_url(str(deepgram_config.api_base))
    assert "encoding=linear16" in deepgram.build_ws_url(str(deepgram_config.api_base))
    assert "speed=" not in deepgram.build_ws_url(str(deepgram_config.api_base))
    assert deepgram.headers() == {"Authorization": "Token deepgram-key"}

    aura_config = StreamingTTSClientConfig(
        provider="deepgram_aura",
        api_base="https://api.deepgram.com",
        api_key="deepgram-key",
        model="aura-2-thalia-en",
    )
    aura = DeepgramAuraStreamingProtocol(aura_config, "deepgram-key")
    assert "/v1/speak?" in aura.build_ws_url(str(aura_config.api_base))
    assert "encoding=linear16" in aura.build_ws_url(str(aura_config.api_base))
    assert "speed=1.0" in aura.build_ws_url(str(aura_config.api_base))
    assert aura.headers() == {"Authorization": "Token deepgram-key"}

    cartesia_config = StreamingTTSClientConfig(
        provider="cartesia",
        api_base="https://api.cartesia.ai",
        api_key="cartesia-key",
        model="sonic-3.5",
        voice_id="voice/id",
        language="en",
        max_buffer_delay_ms=3000,
    )
    cartesia = CartesiaStreamingProtocol(cartesia_config, "cartesia-key")
    assert (
        cartesia.build_ws_url(str(cartesia_config.api_base))
        == "wss://api.cartesia.ai/tts/websocket"
    )
    assert cartesia.headers() == {
        "X-API-Key": "cartesia-key",
        "Cartesia-Version": "2026-03-01",
    }

    first = json.loads(cartesia.text_message("hello "))
    second = json.loads(cartesia.text_message("world"))
    finish = json.loads(cartesia.finish_messages()[0])
    assert first["model_id"] == "sonic-3.5"
    assert first["transcript"] == "hello "
    assert first["language"] == "en"
    assert first["context_id"] == second["context_id"] == finish["context_id"]
    assert first["continue"] is True
    assert second["continue"] is True
    assert finish["continue"] is False
    assert finish["transcript"] == ""
    assert first["output_format"] == {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": 24000,
    }
    assert first["max_buffer_delay_ms"] == 3000

    audio = base64.b64encode(b"pcm").decode("ascii")
    chunk = cartesia.parse(json.dumps({"type": "chunk", "data": audio}))
    # Audio is carried encoded and decoded in bulk once the stream ends.
    assert chunk.audio == audio
    assert chunk.audio_nbytes == 3
    assert chunk.response_started
    malformed_chunk = cartesia.parse(json.dumps({"type": "chunk"}))
    assert malformed_chunk.error == "Cartesia chunk omitted audio data"
    error = cartesia.parse(
        json.dumps({"type": "error", "error_code": "bad_context"})
    )
    assert error.error == "bad_context"
    done = cartesia.parse(json.dumps({"type": "done", "done": True}))
    assert done.audio_done
    assert done.terminal

    another_context = CartesiaStreamingProtocol(
        cartesia_config, "cartesia-key"
    ).context_id
    assert another_context != cartesia.context_id


@pytest.mark.unit
def test_raw_pcm_protocols_carry_and_join_audio_as_bytes() -> None:
    config = StreamingTTSClientConfig(
        provider="vajra",
        api_base="http://example.test",
        model="tts-1",
        pacing=_pacing(),
    )
    frames = [b"\x00\x01" * 8, b"\x02\x03" * 4]
    vajra = VajraStreamingProtocol(config, None)

    events = [vajra.parse(frame) for frame in frames]
    assert [event.audio for event in events] == frames
    assert [event.audio_nbytes for event in events] == [len(f) for f in frames]

    content, error = RawAudioEvent.join_audio([event.audio for event in events])
    assert content == b"".join(frames)
    assert error is None


@pytest.mark.unit
def test_cartesia_client_uses_a_fresh_context_for_each_request() -> None:
    config = StreamingTTSClientConfig(
        provider="cartesia",
        api_base="https://api.cartesia.ai",
        api_key="test-key",
        model="sonic-3.5",
        voice_id="test-voice",
        language="en",
        max_buffer_delay_ms=3000,
        pacing=_pacing(),
    )
    client = StreamingTTSClient(config)
    websockets: list[_FakeWebSocket] = []

    def connect_factory(*_args, **_kwargs) -> _FakeConnection:
        websocket = _FakeWebSocket("cartesia")
        websockets.append(websocket)
        return _FakeConnection(websocket)

    client._connect = connect_factory  # type: ignore[method-assign]
    first_result = asyncio.run(client.send_request(_request(), session_id=1))
    second_result = asyncio.run(client.send_request(_request(), session_id=2))

    assert first_result.success
    assert second_result.success
    context_ids = {
        json.loads(websocket.sent[0])["context_id"] for websocket in websockets
    }
    assert len(context_ids) == 2


@pytest.mark.unit
def test_deepgram_aura_adapter_streams_audio_before_flush_completion() -> None:
    config = StreamingTTSClientConfig(
        provider="deepgram_aura",
        api_base="https://api.deepgram.com",
        api_key="test-key",
        model="aura-2-thalia-en",
        pacing=_pacing(),
    )
    client = StreamingTTSClient(config)
    websocket = _FakeWebSocket("deepgram_aura")
    client._connect = lambda *_args, **_kwargs: _FakeConnection(websocket)  # type: ignore[method-assign]

    result = asyncio.run(client.send_request(_request(), session_id=1))

    assert result.success
    metrics = result.channels[ChannelModality.AUDIO].metrics
    assert metrics[AudioMetricKey.PROVIDER_PROTOCOL.value] == "v1_aura_speak"
    first_audio_ms = metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value][0][0]
    commit_ms = metrics[AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value]
    trigger_ms = metrics[AudioMetricKey.RESPONSE_TRIGGER_OFFSET_MS.value]
    assert trigger_ms < commit_ms
    assert first_audio_ms < commit_ms
    assert metrics[AudioMetricKey.TTFC.value] == pytest.approx(
        first_audio_ms - trigger_ms,
        abs=1.0,
    )
    assert [json.loads(raw)["type"] for raw in websocket.sent] == [
        "Speak",
        "Speak",
        "Speak",
        "Flush",
    ]


@pytest.mark.unit
def test_streaming_tts_rejects_terminal_response_without_audio() -> None:
    config = StreamingTTSClientConfig(
        provider="elevenlabs",
        api_base="https://api.elevenlabs.io",
        api_key="test-key",
        model="eleven_flash_v2_5",
        voice_id="test-voice",
        pacing=_pacing(),
    )
    client = StreamingTTSClient(config)
    websocket = _SilentElevenLabsWebSocket()
    client._connect = lambda *_args, **_kwargs: _FakeConnection(websocket)  # type: ignore[method-assign]

    result = asyncio.run(client.send_request(_request(), session_id=1))

    assert not result.success
    assert result.error_code == 502
    assert result.error_msg == "elevenlabs completed the TTS stream without audio"


@pytest.mark.unit
def test_streaming_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    config = StreamingTTSClientConfig(
        provider="deepgram_flux",
        api_base="https://api.deepgram.com",
        model="flux-alexis-en",
    )
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        StreamingTTSClient(config)


@pytest.mark.unit
def test_streaming_error_priority_preserves_provider_error() -> None:
    provider_error = StreamingTTSError("provider failed")
    grouped = ExceptionGroup(
        "task failures",
        [OSError("socket closed"), ExceptionGroup("provider", [provider_error])],
    )

    flattened = flatten_ws_exception(grouped, _ERROR_PRIORITY)

    assert flattened is provider_error
    assert _map_error(flattened) == (500, "provider failed")
