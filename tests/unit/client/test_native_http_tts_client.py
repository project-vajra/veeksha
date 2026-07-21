from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from typing import Any

import httpx
import pytest

if (
    "transformers" not in sys.modules
    and importlib.util.find_spec("transformers") is None
):
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = object  # type: ignore[attr-defined]
    sys.modules["transformers"] = transformers_stub

from veeksha.client.native_http_tts import (
    DeepgramFluxHTTPClient,
    DeepgramFluxHTTPProtocol,
    ElevenLabsHTTPProtocol,
    ElevenLabsHTTPTTSClient,
)
from veeksha.config.client import (
    DeepgramFluxHTTPClientConfig,
    ElevenLabsHTTPTTSClientConfig,
)
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.types import ChannelModality


class _FakeAsyncClient:
    def __init__(
        self,
        response: httpx.Response,
    ) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        return self.response


def _request() -> Request:
    return Request(
        id=17,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text="A real complete-text TTS request."
            )
        },
    )


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["elevenlabs", "deepgram"])
def test_native_http_clients_buffer_complete_audio_response(provider: str) -> None:
    if provider == "elevenlabs":
        config = ElevenLabsHTTPTTSClientConfig(
            api_base="https://api.elevenlabs.io",
            api_key="test-key",
            model="eleven_flash_v2_5",
            voice_id="test-voice",
        )
        client: Any = ElevenLabsHTTPTTSClient(config)
    else:
        config = DeepgramFluxHTTPClientConfig(
            api_base="https://api.deepgram.com",
            api_key="test-key",
            model="flux-alexis-en",
        )
        client = DeepgramFluxHTTPClient(config)

    request = httpx.Request("POST", "https://provider.example/speak")
    response = httpx.Response(
        200,
        content=bytes(4_800),
        headers={"Content-Type": "audio/l16"},
        request=request,
    )
    fake_client = _FakeAsyncClient(response)
    client._get_client = lambda: fake_client

    sent_count = 0

    def on_sent() -> None:
        nonlocal sent_count
        sent_count += 1

    result = asyncio.run(
        client.send_request(_request(), session_id=1, on_request_sent=on_sent)
    )

    assert result.success
    assert sent_count == 1
    assert len(fake_client.calls) == 1
    metrics = result.channels[ChannelModality.AUDIO].metrics
    assert metrics[AudioMetricKey.PROVIDER.value] == provider
    assert metrics[AudioMetricKey.PROVIDER_MODEL.value] == config.model
    assert metrics[AudioMetricKey.RAW_PCM.value]
    assert metrics[AudioMetricKey.CHUNK_COUNT.value] == 1
    assert (
        metrics[AudioMetricKey.TTFC.value]
        == metrics[AudioMetricKey.END_TO_END_LATENCY.value]
    )
    assert metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value] == [
        [metrics[AudioMetricKey.TTFC.value], 4_800]
    ]


@pytest.mark.unit
def test_native_http_protocols_use_raw_pcm_and_provider_auth() -> None:
    eleven_config = ElevenLabsHTTPTTSClientConfig(
        api_base="https://api.elevenlabs.io",
        api_key="eleven-key",
        model="eleven_flash_v2_5",
        voice_id="voice/id",
    )
    eleven = ElevenLabsHTTPProtocol(eleven_config, "eleven-key")
    eleven_request = eleven.build_request(str(eleven_config.api_base), "hello")
    assert "/voice%2Fid?" in eleven_request.url
    assert "output_format=pcm_24000" in eleven_request.url
    assert eleven_request.headers["xi-api-key"] == "eleven-key"
    assert eleven_request.payload["text"] == "hello"

    deepgram_config = DeepgramFluxHTTPClientConfig(
        api_base="https://api.deepgram.com",
        api_key="deepgram-key",
        model="flux-alexis-en",
    )
    deepgram = DeepgramFluxHTTPProtocol(deepgram_config, "deepgram-key")
    deepgram_request = deepgram.build_request(str(deepgram_config.api_base), "hello")
    assert "/v2/speak?" in deepgram_request.url
    assert "encoding=linear16" in deepgram_request.url
    assert "container=none" in deepgram_request.url
    assert deepgram_request.headers["Authorization"] == "Token deepgram-key"


@pytest.mark.unit
def test_native_http_client_preserves_provider_error() -> None:
    config = DeepgramFluxHTTPClientConfig(
        api_base="https://api.deepgram.com",
        api_key="test-key",
        model="flux-alexis-en",
    )
    client: Any = DeepgramFluxHTTPClient(config)
    request = httpx.Request("POST", "https://provider.example/speak")
    client._get_client = lambda: _FakeAsyncClient(
        httpx.Response(429, text="rate limited", request=request)
    )

    result = asyncio.run(client.send_request(_request(), session_id=1))

    assert not result.success
    assert result.error_code == 429
    assert result.channels == {}
