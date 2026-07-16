import asyncio
import json

import pytest
import websockets

from veeksha.client.stt import (
    VajraOpenAIRealtimeSTTClient,
    VllmRealtimeSTTClient,
    _slice_pcm16_bytes,
)
from veeksha.config.client import STTClientConfig
from veeksha.core.request import Request


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


def _vajra_realtime_client() -> VajraOpenAIRealtimeSTTClient:
    config = STTClientConfig(
        provider="vajra_openai_realtime",
        model="mistralai/Voxtral-Mini-4B-Realtime-2602",
        api_base="http://localhost:8003",
    )
    return VajraOpenAIRealtimeSTTClient(config)


@pytest.mark.unit
def test_vajra_openai_realtime_parses_transcription_events() -> None:
    client = _vajra_realtime_client()

    assert client._parse_message(
        {"type": "conversation.item.input_audio_transcription.delta", "delta": "hi"}
    ) == ("delta", "hi")
    assert client._parse_message(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "hi there",
        }
    ) == ("done", "hi there")
    assert client._parse_message({"type": "input_audio_buffer.committed"}) == ("", "")


@pytest.mark.unit
def test_vajra_openai_realtime_maps_failed_item_to_error() -> None:
    client = _vajra_realtime_client()

    kind, text = client._parse_message(
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "error": {"type": "server_error", "message": "boom"},
        }
    )

    assert (kind, text) == ("error", "boom")


@pytest.mark.unit
def test_vajra_openai_realtime_maps_session_error() -> None:
    client = _vajra_realtime_client()

    assert client._parse_message(
        {"type": "error", "error": {"message": "bad request"}}
    ) == ("error", "bad request")


def _vllm_realtime_client() -> VllmRealtimeSTTClient:
    config = STTClientConfig(
        provider="vllm_realtime",
        model="mistralai/Voxtral-Mini-4B-Realtime-2602",
        api_base="http://localhost:8025",
    )
    return VllmRealtimeSTTClient(config)


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
def test_vllm_stream_fires_callbacks_after_handshake_and_first_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _vllm_realtime_client()
    websocket = _FakeVllmWebSocket()
    monkeypatch.setattr(
        websockets,
        "connect",
        lambda *_args, **_kwargs: _FakeConnection(websocket),
    )
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
