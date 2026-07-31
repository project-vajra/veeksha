from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
import types
from typing import Any

import pytest

# The focused free-threaded unit-test environment omits the heavyweight
# tokenizer dependency. Realtime TTS uses the built-in word tokenizer, but the
# client package imports the generic tokenizer module during initialization.
if (
    "transformers" not in sys.modules
    and importlib.util.find_spec("transformers") is None
):
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = object  # type: ignore[attr-defined]
    sys.modules["transformers"] = transformers_stub

from veeksha.client.streaming_tts import StreamingTTSClient, _b64_decoded_size
from veeksha.config.client import StreamingTTSClientConfig, TextPacingConfig
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.types import ChannelModality


class _FakeWebSocket:
    def __init__(self, deltas: list[str] | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self._events: asyncio.Queue[str] = asyncio.Queue()
        self._deltas = (
            deltas
            if deltas is not None
            else [base64.b64encode(bytes(960)).decode("ascii")]
        )

    async def send(self, raw: str) -> None:
        event = json.loads(raw)
        self.sent.append(event)
        if event["type"] == "session.update":
            await self._events.put(
                json.dumps(
                    {
                        "type": "session.updated",
                        "session": {
                            "audio": {
                                "output": {
                                    "format": {"type": "audio/pcm", "rate": 24000}
                                }
                            }
                        },
                    }
                )
            )
        elif event["type"] == "response.create":
            response_events: list[dict[str, Any]] = [{"type": "response.created"}]
            response_events += [
                {"type": "response.output_audio.delta", "delta": delta}
                for delta in self._deltas
            ]
            response_events += [
                {"type": "response.output_audio.done"},
                {"type": "response.done", "response": {"status": "completed"}},
            ]
            for response_event in response_events:
                await self._events.put(json.dumps(response_event))

    async def recv(self) -> str:
        return await self._events.get()


class _FakeConnection:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> _FakeWebSocket:
        return self.websocket

    async def __aexit__(self, *args: object) -> None:
        return None


def _request() -> Request:
    return Request(
        id=7,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(input_text="one two three")
        },
    )


def _client(
    mode: str, deltas: list[str] | None = None
) -> tuple[StreamingTTSClient, _FakeWebSocket]:
    config = StreamingTTSClientConfig(
        provider="openai_realtime",
        api_base="http://example.test",
        model="test-tts",
        input_output_mode=mode,
        duplex_start_after_tokens=1,
        pacing=TextPacingConfig(
            tokens_per_second=1000.0,
            tokens_per_delta=1,
        ),
    )
    client = StreamingTTSClient(config)
    websocket = _FakeWebSocket(deltas)
    client._connect = lambda *_args, **_kwargs: _FakeConnection(websocket)  # type: ignore[method-assign]
    return client, websocket


@pytest.mark.unit
def test_duplex_triggers_response_before_input_is_complete() -> None:
    client, websocket = _client("duplex")

    result = asyncio.run(client.send_request(_request(), session_id=1))

    event_types = [event["type"] for event in websocket.sent]
    assert event_types == [
        "session.update",
        "conversation.item.create",
        "response.create",
        "conversation.item.create",
        "conversation.item.create",
    ]
    metrics = result.channels[ChannelModality.AUDIO].metrics
    assert (
        metrics[AudioMetricKey.RESPONSE_TRIGGER_OFFSET_MS.value]
        < metrics[AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value]
    )
    assert (
        metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value][0][0]
        < metrics[AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value]
    )


@pytest.mark.unit
def test_complete_text_triggers_response_after_last_input_delta() -> None:
    client, websocket = _client("complete_text")

    result = asyncio.run(client.send_request(_request(), session_id=1))

    event_types = [event["type"] for event in websocket.sent]
    assert event_types == [
        "session.update",
        "conversation.item.create",
        "conversation.item.create",
        "conversation.item.create",
        "response.create",
    ]
    metrics = result.channels[ChannelModality.AUDIO].metrics
    assert (
        metrics[AudioMetricKey.RESPONSE_TRIGGER_OFFSET_MS.value]
        >= metrics[AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value]
    )


@pytest.mark.unit
@pytest.mark.parametrize("n_bytes", [0, 1, 2, 3, 4, 5, 639, 640, 641, 960, 1999])
def test_b64_decoded_size_matches_a_real_decode(n_bytes: int) -> None:
    """The recv loop reports chunk sizes from this without decoding."""
    encoded = base64.b64encode(bytes(n_bytes)).decode("ascii")
    assert _b64_decoded_size(encoded) == len(base64.b64decode(encoded))


@pytest.mark.unit
def test_deferred_decode_yields_the_same_bytes_as_an_inline_decode() -> None:
    chunks = [bytes(960), bytes(480)]
    deltas = [base64.b64encode(chunk).decode("ascii") for chunk in chunks]
    client, _ = _client("complete_text", deltas)

    result = asyncio.run(client.send_request(_request(), session_id=1))

    channel = result.channels[ChannelModality.AUDIO]
    assert result.success
    assert channel.content == b"".join(chunks)
    assert channel.metrics[AudioMetricKey.CHUNK_COUNT.value] == 2
    sizes = [
        size
        for _offset, size in channel.metrics[
            AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value
        ]
    ]
    assert sizes == [960, 480]


@pytest.mark.unit
def test_malformed_base64_fails_the_request_but_keeps_earlier_audio() -> None:
    """Decoding now happens after the stream, so a bad chunk fails late.

    The observable outcome is unchanged from the inline decode: the request
    fails with 500 and the audio received before the bad chunk survives.
    """
    good = base64.b64encode(bytes(960)).decode("ascii")
    client, _ = _client("complete_text", [good, "!!!!"])

    result = asyncio.run(client.send_request(_request(), session_id=1))

    assert not result.success
    assert result.error_code == 500
    assert "Invalid base64 audio" in (result.error_msg or "")
    assert result.channels[ChannelModality.AUDIO].content == bytes(960)


@pytest.mark.unit
def test_realtime_tts_config_rejects_unknown_input_output_mode() -> None:
    with pytest.raises(ValueError, match="input_output_mode"):
        StreamingTTSClientConfig(
            provider="openai_realtime",
            api_base="http://example.test",
            model="test-tts",
            input_output_mode="not-a-mode",
        )
