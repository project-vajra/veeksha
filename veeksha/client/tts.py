"""Provider-agnostic HTTP text-to-speech client."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote, urlencode, urljoin

import httpx

from veeksha.client.base import BaseLLMClient
from veeksha.client.utils import resolve_provider_api_key
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import TTSClientConfig

logger = init_logger(__name__)

__all__ = ["TTSClient"]


@dataclass(frozen=True)
class TTSHTTPRequest:
    """Normalized request emitted by an HTTP provider strategy."""

    url: str
    headers: dict[str, str]
    payload: dict[str, Any]


class TTSProtocolError(Exception):
    """Raised when an HTTP TTS response violates the audio contract."""


class HTTPProviderProtocol(Protocol):
    """Provider-specific request construction behind the shared HTTP lifecycle."""

    provider: str
    protocol_name: str
    default_api_key_env: str
    requires_api_key: bool
    raw_pcm: bool

    def __init__(self, config: TTSClientConfig, api_key: str | None) -> None: ...

    def build_request(self, api_base: str, text: str) -> TTSHTTPRequest: ...


def _audio_speech_url(api_base: str) -> str:
    normalized_base = api_base.rstrip("/")
    if not normalized_base.endswith("/v1"):
        normalized_base = f"{normalized_base}/v1"
    return urljoin(f"{normalized_base}/", "audio/speech")


class OpenAIHTTPProtocol:
    provider = "openai"
    protocol_name = "v1_audio_speech"
    default_api_key_env = "OPENAI_API_KEY"
    requires_api_key = False

    def __init__(self, config: TTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key
        self.raw_pcm = config.raw_pcm

    def build_request(self, api_base: str, text: str) -> TTSHTTPRequest:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return TTSHTTPRequest(
            url=_audio_speech_url(api_base),
            headers=headers,
            payload={
                "model": self.config.model,
                "input": text,
                "voice": self.config.voice_id,
                "response_format": "pcm" if self.config.raw_pcm else "wav",
                "stream": True,
                "stream_format": "audio",
            },
        )


class ElevenLabsHTTPProtocol:
    provider = "elevenlabs"
    protocol_name = "v1_text_to_speech"
    default_api_key_env = "ELEVENLABS_API_KEY"
    requires_api_key = True
    raw_pcm = True

    def __init__(self, config: TTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def build_request(self, api_base: str, text: str) -> TTSHTTPRequest:
        normalized = api_base.rstrip("/") + "/"
        path = f"v1/text-to-speech/{quote(self.config.voice_id, safe='')}"
        query = urlencode({"output_format": f"pcm_{self.config.sample_rate}"})
        return TTSHTTPRequest(
            url=f"{urljoin(normalized, path)}?{query}",
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            payload={
                "text": text,
                "model_id": self.config.model,
                "voice_settings": {
                    "stability": self.config.stability,
                    "similarity_boost": self.config.similarity_boost,
                    "speed": self.config.speed,
                },
                "apply_text_normalization": self.config.apply_text_normalization,
            },
        )


class DeepgramFluxHTTPProtocol:
    provider = "deepgram"
    protocol_name = "v2_flux_speak_http"
    default_api_key_env = "DEEPGRAM_API_KEY"
    requires_api_key = True
    raw_pcm = True

    def __init__(self, config: TTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def build_request(self, api_base: str, text: str) -> TTSHTTPRequest:
        normalized = api_base.rstrip("/") + "/"
        query = urlencode(
            {
                "model": self.config.model,
                "encoding": "linear16",
                "container": "none",
                "sample_rate": self.config.sample_rate,
                "mip_opt_out": str(self.config.mip_opt_out).lower(),
            }
        )
        return TTSHTTPRequest(
            url=f"{urljoin(normalized, 'v2/speak')}?{query}",
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            },
            payload={"text": text},
        )


_HTTP_PROTOCOLS: dict[str, type[HTTPProviderProtocol]] = {
    "openai": OpenAIHTTPProtocol,
    "elevenlabs": ElevenLabsHTTPProtocol,
    "deepgram_flux": DeepgramFluxHTTPProtocol,
}


class TTSClient(BaseLLMClient):
    """Complete-text HTTP TTS client with provider-specific request strategies."""

    def __init__(self, config: TTSClientConfig, **kwargs: Any) -> None:
        protocol_class = _HTTP_PROTOCOLS.get(config.provider)
        if protocol_class is None:
            raise ValueError(f"Unsupported HTTP TTS provider: {config.provider}")
        super().__init__(config)
        self.api_key = resolve_provider_api_key(
            config.api_key,
            config.api_key_env,
            protocol_class.default_api_key_env,
            required=protocol_class.requires_api_key,
        )
        self._http_config = config
        self._protocol = protocol_class(config, self.api_key)
        self._client_storage = threading.local()

    def _get_client(self) -> httpx.AsyncClient:
        if not hasattr(self._client_storage, "client"):
            self._client_storage.client = httpx.AsyncClient(
                timeout=self._http_config.request_timeout
            )
        return self._client_storage.client

    @staticmethod
    def _validate_audio_response(response: httpx.Response) -> None:
        content_type = response.headers.get("Content-Type", "")
        media_type = content_type.partition(";")[0].strip().lower()
        allowed = {
            "application/octet-stream",
            "application/x-wav",
            "binary/octet-stream",
        }
        if (
            media_type
            and not media_type.startswith("audio/")
            and media_type not in allowed
        ):
            raise TTSProtocolError(
                f"Expected an audio byte response, got Content-Type {content_type!r}"
            )

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent: Callable[[], None] | None = None,
        on_request_dispatched: Callable[[], None] | None = None,
    ) -> RequestResult:
        text_content = request.channels.get(ChannelModality.TEXT)
        if not isinstance(text_content, TextChannelRequestContent):
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=400,
                error_msg="No TEXT channel in request for TTS",
                client_completed_at=time.monotonic(),
            )

        input_text = text_content.input_text
        provider_request = self._protocol.build_request(str(self.api_base), input_text)
        audio_chunks: list[bytes] = []
        audio_chunk_timestamps: list[list[float | int]] = []
        error_code: int | None = None
        error_msg: str | None = None
        ttfc: float | None = None
        sent_fired = False
        start = time.monotonic()

        def fire_sent_once() -> None:
            nonlocal sent_fired
            if not sent_fired and on_request_sent is not None:
                on_request_sent()
                sent_fired = True

        try:
            async with self._get_client().stream(
                "POST",
                provider_request.url,
                headers=provider_request.headers,
                json=provider_request.payload,
                timeout=self._http_config.request_timeout,
            ) as response:
                response.raise_for_status()
                self._validate_audio_response(response)
                if on_request_dispatched is not None:
                    on_request_dispatched()
                async for chunk in response.aiter_bytes(
                    chunk_size=self._http_config.chunk_size
                ):
                    if not chunk:
                        continue
                    offset_ms = (time.monotonic() - start) * 1000
                    if ttfc is None:
                        ttfc = offset_ms
                    audio_chunks.append(chunk)
                    audio_chunk_timestamps.append([round(offset_ms, 3), len(chunk)])
                    fire_sent_once()
                if not audio_chunks:
                    raise TTSProtocolError(
                        f"{self._protocol.provider} returned an empty audio response"
                    )
        except TTSProtocolError as exc:
            error_code = 502
            error_msg = str(exc)
        except httpx.HTTPStatusError as exc:
            error_code = exc.response.status_code
            error_msg = str(exc)
        except httpx.ConnectError as exc:
            error_code = 503
            error_msg = str(exc)
        except httpx.TimeoutException:
            error_code = 408
            error_msg = "TTS request timed out"
        except Exception as exc:
            error_code = 520
            error_msg = str(exc)
            logger.exception(
                "%s HTTP TTS error: (%s) %s",
                self._protocol.provider,
                error_code,
                error_msg,
            )

        completed_at = time.monotonic()
        latency_ms = (completed_at - start) * 1000
        success = error_code is None and error_msg is None
        fire_sent_once()

        if not success:
            logger.warning(
                "%s HTTP TTS error: (%s) %s",
                self._protocol.provider,
                error_code,
                error_msg,
            )

        channels: dict[ChannelModality, ChannelResponse] = {}
        if success:
            rounded_latency = round(latency_ms, 3)
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=b"".join(audio_chunks),
                metrics={
                    "audio_task": AudioTask.TTS,
                    AudioMetricKey.PROVIDER.value: self._protocol.provider,
                    AudioMetricKey.PROVIDER_MODEL.value: self._http_config.model,
                    AudioMetricKey.PROVIDER_PROTOCOL.value: (
                        self._protocol.protocol_name
                    ),
                    AudioMetricKey.TTFC.value: round(ttfc or 0.0, 3),
                    AudioMetricKey.END_TO_END_LATENCY.value: rounded_latency,
                    AudioMetricKey.CHUNK_COUNT.value: len(audio_chunks),
                    AudioMetricKey.RAW_PCM.value: self._protocol.raw_pcm,
                    AudioMetricKey.SAMPLE_RATE.value: self._http_config.sample_rate,
                    AudioMetricKey.INPUT_CHARS.value: len(input_text),
                    AudioMetricKey.INPUT_TOKENS.value: (
                        text_content.target_prompt_tokens or 0
                    ),
                    AudioMetricKey.INPUT_TEXT.value: input_text,
                    AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value: [
                        [0.0, len(input_text)]
                    ],
                    AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value: (
                        audio_chunk_timestamps
                    ),
                    AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value: 0.0,
                    AudioMetricKey.AUDIO_DONE_OFFSET_MS.value: rounded_latency,
                    AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value: rounded_latency,
                },
            )

        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            channels=channels,
            success=success,
            error_code=error_code,
            error_msg=error_msg,
            client_completed_at=completed_at,
        )
