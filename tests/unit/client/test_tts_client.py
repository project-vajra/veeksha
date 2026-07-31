from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
import types
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import numpy as np
import pytest
from multidict import CIMultiDict
from yarl import URL

if (
    "transformers" not in sys.modules
    and importlib.util.find_spec("transformers") is None
):
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = object  # type: ignore[attr-defined]
    sys.modules["transformers"] = transformers_stub

from veeksha.client.tts import (
    DeepgramFluxHTTPProtocol,
    ElevenLabsHTTPProtocol,
    MistralHTTPProtocol,
    OpenAIHTTPProtocol,
    TTSClient,
    _float32_pcm_to_pcm16,
)
from veeksha.config.client import TTSClientConfig
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.types import ChannelModality


class _FakeContent:
    """The pieces of ``aiohttp``'s streaming body the TTS protocols use."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def iter_chunked(self, chunk_size: int) -> AsyncIterator[bytes]:
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    async def __aiter__(self) -> AsyncIterator[bytes]:
        # aiohttp iterates a response body line by line, newline included.
        for line in self._body.splitlines(keepends=True):
            yield line


class _FakeResponse:
    def __init__(
        self, status: int, *, body: bytes = b"", headers: dict[str, str] | None = None
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self.content = _FakeContent(body)

    def raise_for_status(self) -> None:
        if self.status >= 400:
            url = URL("https://provider.example/speak")
            raise aiohttp.ClientResponseError(
                request_info=aiohttp.RequestInfo(
                    url=url, method="POST", headers=CIMultiDict(), real_url=url
                ),
                history=(),
                status=self.status,
                message="rate limited",
            )


class _FakeRequestContext:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    async def __aenter__(self) -> _FakeResponse:
        return self.response

    async def __aexit__(self, *args: object) -> None:
        return None


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _FakeRequestContext:
        self.calls.append({"url": url, **kwargs})
        return _FakeRequestContext(self.response)


def _request() -> Request:
    return Request(
        id=17,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text="A real complete-text TTS request."
            )
        },
    )


def _config(provider: str) -> TTSClientConfig:
    kwargs: dict[str, Any] = {
        "provider": provider,
        "api_base": "https://provider.example",
        "api_key": "test-key",
        "chunk_size": 4_800,
    }
    if provider == "openai":
        kwargs.update(model="gpt-4o-mini-tts", voice_id="alloy", raw_pcm=True)
    elif provider == "elevenlabs":
        kwargs.update(model="eleven_flash_v2_5", voice_id="test-voice")
    elif provider == "mistral":
        kwargs.update(model="voxtral-mini-tts-2603", voice_id="test-voice")
    else:
        kwargs.update(model="flux-alexis-en")
    return TTSClientConfig(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "metric_provider"),
    [
        ("openai", "openai"),
        ("elevenlabs", "elevenlabs"),
        ("deepgram_flux", "deepgram"),
        ("mistral", "mistral"),
    ],
)
def test_http_providers_share_streaming_audio_lifecycle(
    provider: str, metric_provider: str
) -> None:
    config = _config(provider)
    client = TTSClient(config)
    if provider == "mistral":
        float32_audio = np.zeros(2_400, dtype="<f4").tobytes()
        encoded_audio = base64.b64encode(float32_audio).decode("ascii")
        content = (
            "event: speech.audio.delta\n"
            f"data: {json.dumps({'audio_data': encoded_audio})}\n\n"
            "event: speech.audio.done\n"
            "data: {}\n\n"
        ).encode()
        response = _FakeResponse(
            200,
            body=content,
            headers={"Content-Type": "text/event-stream"},
        )
    else:
        response = _FakeResponse(
            200,
            body=bytes(4_800),
            headers={"Content-Type": "audio/l16"},
        )
    fake_client = _FakeAsyncClient(response)
    client._get_client = lambda: fake_client  # type: ignore[method-assign]
    events: list[str] = []

    result = asyncio.run(
        client.send_request(
            _request(),
            session_id=1,
            on_request_dispatched=lambda: events.append("dispatched"),
            on_request_sent=lambda: events.append("sent"),
        )
    )

    assert result.success
    assert events == ["dispatched", "sent"]
    assert len(fake_client.calls) == 1
    metrics = result.channels[ChannelModality.AUDIO].metrics
    assert metrics[AudioMetricKey.PROVIDER.value] == metric_provider
    assert metrics[AudioMetricKey.PROVIDER_MODEL.value] == config.model
    assert metrics[AudioMetricKey.RAW_PCM.value]
    assert metrics[AudioMetricKey.CHUNK_COUNT.value] == 1
    assert metrics[AudioMetricKey.RESPONSE_TRIGGER_OFFSET_MS.value] == 0.0
    assert (
        metrics[AudioMetricKey.TTFC.value]
        <= metrics[AudioMetricKey.END_TO_END_LATENCY.value]
    )
    assert metrics[AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value][0][1] == 4_800


@pytest.mark.unit
def test_http_protocols_keep_provider_specific_url_payload_and_auth() -> None:
    openai_config = _config("openai")
    openai = OpenAIHTTPProtocol(openai_config, "openai-key")
    openai_request = openai.build_request(str(openai_config.api_base), "hello")
    assert openai_request.url.endswith("/v1/audio/speech")
    assert openai_request.headers["Authorization"] == "Bearer openai-key"
    assert openai_request.payload["voice"] == "alloy"

    eleven_config = TTSClientConfig(
        provider="elevenlabs",
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

    deepgram_config = _config("deepgram_flux")
    deepgram = DeepgramFluxHTTPProtocol(deepgram_config, "deepgram-key")
    deepgram_request = deepgram.build_request(str(deepgram_config.api_base), "hello")
    assert "/v2/speak?" in deepgram_request.url
    assert "encoding=linear16" in deepgram_request.url
    assert "container=none" in deepgram_request.url
    assert deepgram_request.headers["Authorization"] == "Token deepgram-key"

    mistral_config = _config("mistral")
    mistral = MistralHTTPProtocol(mistral_config, "mistral-key")
    mistral_request = mistral.build_request(str(mistral_config.api_base), "hello")
    assert mistral_request.url.endswith("/v1/audio/speech")
    assert mistral_request.headers["Authorization"] == "Bearer mistral-key"
    assert mistral_request.headers["Accept"] == "text/event-stream"
    assert mistral_request.payload["response_format"] == "pcm"
    assert mistral_request.payload["stream"] is True


@pytest.mark.unit
def test_mistral_float32_pcm_is_normalized_to_pcm16() -> None:
    samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype="<f4")

    pcm16 = np.frombuffer(_float32_pcm_to_pcm16(samples.tobytes()), dtype="<i2")

    assert pcm16.tolist() == [-32767, -16383, 0, 16383, 32767]


@pytest.mark.unit
def test_http_client_preserves_provider_status_code() -> None:
    client = TTSClient(_config("deepgram_flux"))
    client._get_client = lambda: _FakeAsyncClient(  # type: ignore[method-assign]
        _FakeResponse(429)
    )

    result = asyncio.run(client.send_request(_request(), session_id=1))

    assert not result.success
    assert result.error_code == 429
    assert result.channels == {}


@pytest.mark.unit
def test_http_cloud_provider_requires_its_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    config = TTSClientConfig(
        provider="deepgram_flux",
        api_base="https://api.deepgram.com",
        model="flux-alexis-en",
    )

    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        TTSClient(config)
