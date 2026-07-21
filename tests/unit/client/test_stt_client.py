import asyncio
import json
from pathlib import Path

import pytest

from veeksha.client import STTClient
from veeksha.client.registry import ClientRegistry
from veeksha.client.stt import (
    _ClipAssets,
    _map_stt_error,
    _slice_pcm16_bytes,
    _STTProtocolError,
)
from veeksha.config.client import STTClientConfig
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import AudioChannelRequestContent
from veeksha.types import ChannelModality, ClientType


@pytest.mark.unit
def test_slice_pcm16_bytes_uses_millisecond_offsets() -> None:
    pcm = bytes(range(20))

    sliced = _slice_pcm16_bytes(pcm, 1000, start_ms=2.0, end_ms=6.0)

    assert sliced == pcm[4:12]


@pytest.mark.unit
def test_slice_pcm16_bytes_passthrough_without_offsets() -> None:
    pcm = bytes(range(20))

    assert _slice_pcm16_bytes(pcm, 1000, start_ms=None, end_ms=None) is pcm


@pytest.mark.unit
def test_slice_pcm16_bytes_open_ended_slices() -> None:
    pcm = bytes(range(20))

    assert _slice_pcm16_bytes(pcm, 1000, start_ms=None, end_ms=6.0) == pcm[:12]
    assert _slice_pcm16_bytes(pcm, 1000, start_ms=2.0, end_ms=None) == pcm[4:]


@pytest.mark.unit
def test_slice_pcm16_bytes_rejects_negative_start() -> None:
    pcm = bytes(range(20))

    with pytest.raises(ValueError, match="must be non-negative"):
        _slice_pcm16_bytes(pcm, 1000, start_ms=-1.0, end_ms=6.0)


@pytest.mark.unit
def test_slice_pcm16_bytes_rejects_end_not_after_start() -> None:
    pcm = bytes(range(20))

    with pytest.raises(ValueError, match="must be greater than"):
        _slice_pcm16_bytes(pcm, 1000, start_ms=6.0, end_ms=6.0)


@pytest.mark.unit
def test_slice_pcm16_bytes_rejects_end_past_clip() -> None:
    pcm = bytes(range(20))

    with pytest.raises(ValueError, match="exceeds decoded clip length"):
        _slice_pcm16_bytes(pcm, 1000, start_ms=0.0, end_ms=11.0)


@pytest.mark.unit
@pytest.mark.parametrize("field_name", ["ws_ping_interval_s", "ws_ping_timeout_s"])
def test_stt_client_config_rejects_nonpositive_ws_ping(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be > 0 or None"):
        STTClientConfig(
            provider="vajra_openai_realtime",
            model="mistralai/Voxtral-Mini-4B-Realtime-2602",
            api_base="http://localhost:8003",
            **{field_name: 0},
        )


@pytest.mark.unit
def test_stt_errors_use_shared_websocket_mapping() -> None:
    assert _map_stt_error(_STTProtocolError("provider failed"), 3.0) == (
        500,
        "provider failed",
    )
    assert _map_stt_error(TimeoutError(), 3.0) == (
        408,
        "STT request timed out after 3.0s",
    )
    assert _map_stt_error(OSError("unreachable"), 3.0) == (503, "unreachable")


def _vajra_realtime_client() -> STTClient:
    config = STTClientConfig(
        provider="vajra_openai_realtime",
        model="mistralai/Voxtral-Mini-4B-Realtime-2602",
        api_base="http://localhost:8003",
    )
    return STTClient(config)


@pytest.mark.unit
def test_vajra_openai_realtime_parses_transcription_events() -> None:
    client = _vajra_realtime_client()

    assert client._protocol.parse_message(
        {"type": "conversation.item.input_audio_transcription.delta", "delta": "hi"}
    ) == ("delta", "hi")
    assert client._protocol.parse_message(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "hi there",
        }
    ) == ("done", "hi there")
    assert client._protocol.parse_message({"type": "input_audio_buffer.committed"}) == (
        "",
        "",
    )


@pytest.mark.unit
def test_vajra_openai_realtime_maps_failed_item_to_error() -> None:
    client = _vajra_realtime_client()

    kind, text = client._protocol.parse_message(
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "error": {"type": "server_error", "message": "boom"},
        }
    )

    assert (kind, text) == ("error", "boom")


@pytest.mark.unit
def test_vajra_openai_realtime_maps_session_error() -> None:
    client = _vajra_realtime_client()

    assert client._protocol.parse_message(
        {"type": "error", "error": {"message": "bad request"}}
    ) == ("error", "bad request")


def _vllm_realtime_client() -> STTClient:
    config = STTClientConfig(
        provider="vllm_realtime",
        model="mistralai/Voxtral-Mini-4B-Realtime-2602",
        api_base="http://localhost:8025",
    )
    return STTClient(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "provider",
        "api_base",
        "metric_provider",
        "protocol_name",
        "expected_ws_url",
    ),
    [
        (
            "vllm_realtime",
            "http://localhost:8025",
            "vllm",
            "v1_realtime_transcription",
            "ws://localhost:8025/v1/realtime",
        ),
        (
            "vajra_openai_realtime",
            "https://localhost:8003",
            "vajra",
            "openai_v1_realtime_transcription",
            "wss://localhost:8003/openai/v1/realtime?intent=transcription",
        ),
    ],
)
def test_registry_exposes_one_stt_client_with_provider_strategies(
    provider: str,
    api_base: str,
    metric_provider: str,
    protocol_name: str,
    expected_ws_url: str,
) -> None:
    config = STTClientConfig(
        provider=provider,
        model="mistralai/Voxtral-Mini-4B-Realtime-2602",
        api_base=api_base,
    )

    client = ClientRegistry.get(ClientType.STT, config=config)

    assert type(client) is STTClient
    assert client._protocol.provider == metric_provider
    assert client._protocol.protocol_name == protocol_name
    assert client._ws_url == expected_ws_url


@pytest.mark.unit
def test_stt_send_request_finishes_lifecycle_callbacks_on_invalid_audio() -> None:
    client = _vllm_realtime_client()
    events: list[str] = []

    result = asyncio.run(
        client.send_request(
            Request(id=99, channels={}),
            session_id=7,
            on_request_sent=lambda: events.append("sent"),
            on_request_dispatched=lambda: events.append("dispatched"),
        )
    )

    assert result.success is False
    assert result.error_code == 400
    assert events == ["dispatched", "sent"]


class _FakeVllmWebSocket:
    def __init__(self) -> None:
        self._recv_index = 0
        self._audio_sent = asyncio.Event()

    async def recv(self) -> str:
        if self._recv_index == 0:
            self._recv_index += 1
            return json.dumps({"type": "session.created"})

        await self._audio_sent.wait()
        if self._recv_index == 1:
            self._recv_index += 1
            return json.dumps({"type": "transcription.delta", "delta": "hello"})
        self._recv_index += 1
        return json.dumps({"type": "transcription.done", "text": "hello"})

    async def send(self, message: str | bytes) -> None:
        if isinstance(message, str):
            payload = json.loads(message)
            if payload.get("type") == "input_audio_buffer.append":
                self._audio_sent.set()


class _FakeConnection:
    def __init__(self, websocket: _FakeVllmWebSocket) -> None:
        self._websocket = websocket

    async def __aenter__(self) -> _FakeVllmWebSocket:
        return self._websocket

    async def __aexit__(self, *_args) -> None:
        return None


@pytest.mark.unit
def test_vllm_stream_fires_callbacks_after_handshake_and_first_content() -> None:
    client = _vllm_realtime_client()
    websocket = _FakeVllmWebSocket()
    client._connect = lambda: _FakeConnection(websocket)  # type: ignore[method-assign]
    events: list[str] = []

    result = asyncio.run(
        client._stream(
            b"\x00\x00",
            on_request_sent=lambda: events.append("sent"),
            on_request_dispatched=lambda: events.append("dispatched"),
        )
    )

    assert result.final_transcript == "hello"
    assert events == ["dispatched", "sent"]


@pytest.mark.unit
def test_stt_request_emits_normalized_provider_metadata(tmp_path: Path) -> None:
    client = _vllm_realtime_client()
    websocket = _FakeVllmWebSocket()
    client._connect = lambda: _FakeConnection(websocket)  # type: ignore[method-assign]
    client._clip_assets = lambda _path: _ClipAssets(  # type: ignore[method-assign]
        pcm=b"\x00\x00",
        wire_messages=[],
    )
    audio_file = tmp_path / "request.pcm"
    audio_file.write_bytes(b"\x00\x00")
    request = Request(
        id=101,
        channels={
            ChannelModality.AUDIO: AudioChannelRequestContent(
                input_audio=str(audio_file)
            )
        },
    )

    result = asyncio.run(client.send_request(request, session_id=3))

    assert result.success
    metrics = result.channels[ChannelModality.AUDIO].metrics
    assert metrics[AudioMetricKey.PROVIDER.value] == "vllm"
    assert metrics[AudioMetricKey.PROVIDER_MODEL.value] == (
        "mistralai/Voxtral-Mini-4B-Realtime-2602"
    )
    assert (
        metrics[AudioMetricKey.PROVIDER_PROTOCOL.value] == "v1_realtime_transcription"
    )
    assert metrics[AudioMetricKey.RAW_PCM.value] is True
