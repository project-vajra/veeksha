"""Native complete-text HTTP clients for cloud TTS providers."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.parse import quote, urlencode, urljoin

import httpx

from veeksha.client.base import BaseLLMClient
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import (
        DeepgramFluxHTTPClientConfig,
        ElevenLabsHTTPTTSClientConfig,
    )

logger = init_logger(__name__)


@dataclass(frozen=True)
class NativeHTTPRequest:
    url: str
    headers: dict[str, str]
    payload: dict[str, Any]


class ElevenLabsHTTPProtocol:
    provider = "elevenlabs"
    protocol_name = "v1_text_to_speech"

    def __init__(self, config: "ElevenLabsHTTPTTSClientConfig", api_key: str) -> None:
        self.config = config
        self.api_key = api_key

    def build_request(self, api_base: str, text: str) -> NativeHTTPRequest:
        normalized = api_base.rstrip("/") + "/"
        path = f"v1/text-to-speech/{quote(self.config.voice_id, safe='')}"
        query = urlencode({"output_format": f"pcm_{self.config.sample_rate}"})
        return NativeHTTPRequest(
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

    def __init__(self, config: "DeepgramFluxHTTPClientConfig", api_key: str) -> None:
        self.config = config
        self.api_key = api_key

    def build_request(self, api_base: str, text: str) -> NativeHTTPRequest:
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
        return NativeHTTPRequest(
            url=f"{urljoin(normalized, 'v2/speak')}?{query}",
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            },
            payload={"text": text},
        )


class NativeNonStreamingTTSClient(BaseLLMClient):
    """Measure a complete-input/complete-output HTTP TTS operation."""

    def __init__(
        self, config: Any, protocol_factory: Callable[[Any, str], Any]
    ) -> None:
        configured_key = config.api_key
        super().__init__(config)
        api_key = configured_key or os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key is required: set client.api_key or {config.api_key_env}"
            )
        self.api_key = api_key
        self._http_config = config
        self._protocol = protocol_factory(config, api_key)
        self._client_storage = threading.local()

    def _get_client(self) -> httpx.AsyncClient:
        if not hasattr(self._client_storage, "client"):
            self._client_storage.client = httpx.AsyncClient(
                timeout=self._http_config.request_timeout
            )
        return self._client_storage.client

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent: Optional[Callable[[], None]] = None,
        on_request_dispatched: Optional[Callable[[], None]] = None,
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
        native_request = self._protocol.build_request(str(self.api_base), input_text)
        t_start = time.monotonic()
        error_code: Optional[int] = None
        error_msg: Optional[str] = None
        audio_data = b""

        try:
            if on_request_dispatched is not None:
                on_request_dispatched()
            response = await self._get_client().post(
                native_request.url,
                headers=native_request.headers,
                json=native_request.payload,
                timeout=self._http_config.request_timeout,
            )
            response.raise_for_status()
            audio_data = response.content
            content_type = response.headers.get("Content-Type", "")
            if content_type and not content_type.lower().startswith(
                ("audio/", "application/octet-stream", "binary/octet-stream")
            ):
                raise ValueError(f"Expected audio response, got {content_type!r}")
            if not audio_data:
                raise ValueError("TTS provider returned an empty audio response")
            if on_request_sent is not None:
                on_request_sent()
        except httpx.HTTPStatusError as exc:
            error_code = exc.response.status_code
            error_msg = str(exc)
        except httpx.ConnectError as exc:
            error_code = 503
            error_msg = str(exc)
        except httpx.TimeoutException:
            error_code = 408
            error_msg = "TTS request timed out"
        except Exception as exc:  # noqa: BLE001 - converted to request result.
            error_code = 502
            error_msg = str(exc)

        completed_at = time.monotonic()
        total_latency_ms = (completed_at - t_start) * 1000
        success = error_code is None and error_msg is None
        if not success:
            logger.warning(
                "%s non-streaming TTS error: (%s) %s",
                self._protocol.provider,
                error_code,
                error_msg,
            )
            if on_request_sent is not None:
                on_request_sent()

        channels: dict = {}
        if success:
            terminal_ms = round(total_latency_ms, 3)
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=audio_data,
                metrics={
                    "audio_task": AudioTask.TTS,
                    AudioMetricKey.PROVIDER.value: self._protocol.provider,
                    AudioMetricKey.PROVIDER_MODEL.value: self._http_config.model,
                    AudioMetricKey.PROVIDER_PROTOCOL.value: (
                        self._protocol.protocol_name
                    ),
                    AudioMetricKey.TTFC.value: terminal_ms,
                    AudioMetricKey.END_TO_END_LATENCY.value: terminal_ms,
                    AudioMetricKey.CHUNK_COUNT.value: 1,
                    AudioMetricKey.RAW_PCM.value: True,
                    AudioMetricKey.SAMPLE_RATE.value: self._http_config.sample_rate,
                    AudioMetricKey.INPUT_CHARS.value: len(input_text),
                    AudioMetricKey.INPUT_TOKENS.value: (
                        text_content.target_prompt_tokens or 0
                    ),
                    AudioMetricKey.INPUT_TEXT.value: input_text,
                    AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value: [
                        [0.0, len(input_text)]
                    ],
                    AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value: [
                        [terminal_ms, len(audio_data)]
                    ],
                    AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value: 0.0,
                    AudioMetricKey.AUDIO_DONE_OFFSET_MS.value: terminal_ms,
                    AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value: terminal_ms,
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


class ElevenLabsHTTPTTSClient(NativeNonStreamingTTSClient):
    def __init__(self, config: "ElevenLabsHTTPTTSClientConfig", **kwargs) -> None:
        super().__init__(config, ElevenLabsHTTPProtocol)


class DeepgramFluxHTTPClient(NativeNonStreamingTTSClient):
    def __init__(self, config: "DeepgramFluxHTTPClientConfig", **kwargs) -> None:
        super().__init__(config, DeepgramFluxHTTPProtocol)
