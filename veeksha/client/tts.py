"""TTS client for streaming HTTP-based text-to-speech APIs."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional
from urllib.parse import urljoin

import httpx

from veeksha.client.base import BaseLLMClient
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import TTSClientConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class TTSProviderRequest:
    url: str
    headers: dict[str, str]
    payload: dict


class TTSProviderAdapter(ABC):
    """Provider-specific request adapter for streaming TTS endpoints."""

    def __init__(self, config: TTSClientConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def raw_pcm(self) -> bool:
        """Whether the provider streams raw PCM bytes."""

    @abstractmethod
    def build_request(self, text: str) -> TTSProviderRequest:
        """Build a streaming HTTP request for a text input."""

    @staticmethod
    def _join_url(api_base: str, endpoint: str) -> str:
        return urljoin(api_base.rstrip("/") + "/", endpoint.lstrip("/"))


class VajraTTSProviderAdapter(TTSProviderAdapter):
    @property
    def raw_pcm(self) -> bool:
        return self.config.raw_pcm

    def build_request(self, text: str) -> TTSProviderRequest:
        return TTSProviderRequest(
            url=self._join_url(str(self.config.api_base), "synthesize/stream"),
            headers={"Content-Type": "application/json"},
            payload={"text": text},
        )


class VLLMOmniTTSProviderAdapter(TTSProviderAdapter):
    @property
    def raw_pcm(self) -> bool:
        return True

    def build_request(self, text: str) -> TTSProviderRequest:
        api_base = str(self.config.api_base)
        endpoint = (
            "audio/speech"
            if api_base.rstrip("/").endswith("/v1")
            else "v1/audio/speech"
        )
        payload: dict = {
            "input": text,
            "model": self.config.model,
            "response_format": "pcm",
            "stream": True,
        }
        if self.config.voice_id:
            payload["voice"] = self.config.voice_id
        return TTSProviderRequest(
            url=self._join_url(api_base, endpoint),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )


def _build_provider_adapter(config: TTSClientConfig) -> TTSProviderAdapter:
    if config.provider == "vajra":
        return VajraTTSProviderAdapter(config)
    if config.provider == "vllm_omni":
        return VLLMOmniTTSProviderAdapter(config)
    raise ValueError(f"Unsupported TTS provider: {config.provider}")


class TTSClient(BaseLLMClient):
    """Async client for streaming HTTP-based TTS APIs."""

    def __init__(self, config: TTSClientConfig, **kwargs) -> None:
        super().__init__(config)
        self._provider = config.provider
        self._chunk_size = config.chunk_size
        self._provider_adapter = _build_provider_adapter(config)
        self._client_storage = threading.local()

    def _get_client(self) -> httpx.AsyncClient:
        """Return a thread-local httpx client bound to the caller's event loop."""
        if not hasattr(self._client_storage, "client"):
            self._client_storage.client = httpx.AsyncClient(
                timeout=self.config.request_timeout
            )
        return self._client_storage.client

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
    ) -> RequestResult:
        """Send a streaming TTS request and collect audio metrics."""
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
        provider_request = self._provider_adapter.build_request(input_text)

        logger.debug(
            "[TTS %s] request_id=%d session_id=%d chars=%d text=%.80r",
            self._provider,
            request.id,
            session_id,
            len(input_text),
            input_text,
        )

        error_msg: Optional[str] = None
        error_code: Optional[int] = None
        ttfc: Optional[float] = None
        chunk_count = 0
        audio_chunks: List[bytes] = []

        t_start = time.monotonic()

        try:
            async with self._get_client().stream(
                "POST",
                provider_request.url,
                headers=provider_request.headers,
                json=provider_request.payload,
                timeout=self.config.request_timeout,
            ) as response:
                response.raise_for_status()

                async for chunk in response.aiter_bytes(chunk_size=self._chunk_size):
                    if chunk:
                        receive_time = time.monotonic()
                        if ttfc is None:
                            ttfc = (receive_time - t_start) * 1000  # ms

                        audio_chunks.append(chunk)
                        chunk_count += 1

        except httpx.HTTPStatusError as e:
            error_code = e.response.status_code if e.response else 500
            error_msg = str(e)
            logger.warning("HTTP Error: status=%s msg=%s", error_code, error_msg)
        except httpx.ConnectError as e:
            error_code = 503
            error_msg = str(e)
            logger.warning("Connection Error: (%s) %s", error_code, error_msg)
        except httpx.TimeoutException:
            error_code = 408
            error_msg = "TTS request timed out"
            logger.warning("Timeout Error: (%s) %s", error_code, error_msg)
        except Exception as e:
            error_code = 520
            error_msg = str(e)
            logger.exception("Unexpected error: (%s) %s", error_code, error_msg)

        completed_at = time.monotonic()
        total_latency_ms = (completed_at - t_start) * 1000
        success = error_msg is None and error_code is None
        audio_data = b"".join(audio_chunks) if audio_chunks else b""

        channels = {}
        if success:
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=audio_data,
                metrics={
                    "audio_task": AudioTask.TTS,
                    "ttfc": round(ttfc or 0.0, 3),
                    "end_to_end_latency": round(total_latency_ms, 3),
                    "chunk_count": chunk_count,
                    "raw_pcm": self._provider_adapter.raw_pcm,
                    "sample_rate": self.config.sample_rate,
                    "input_chars": len(input_text),
                    "input_tokens": text_content.target_prompt_tokens or 0,
                    "input_text": input_text,
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
