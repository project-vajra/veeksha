"""TTS client for streaming HTTP-based text-to-speech APIs."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx

from veeksha.client.base import BaseLLMClient
from veeksha.client.tts_adapters import _build_provider_adapter
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import TTSClientConfig

logger = init_logger(__name__)


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
        on_request_sent: Callable[[], None] | None = None,
        on_request_dispatched: Callable[[], None] | None = None,
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

        error_msg: str | None = None
        error_code: int | None = None
        ttfc: float | None = None
        chunk_count = 0
        audio_chunks: list[bytes] = []

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
                if on_request_dispatched is not None:
                    on_request_dispatched()

                sent_notified = False
                async for chunk in self._provider_adapter.iter_audio_chunks(
                    response, self._chunk_size
                ):
                    receive_time = time.monotonic()
                    if ttfc is None:
                        ttfc = (receive_time - t_start) * 1000
                    if not sent_notified and on_request_sent is not None:
                        on_request_sent()
                        sent_notified = True

                    audio_chunks.append(chunk)
                    chunk_count += 1

                if not sent_notified and on_request_sent is not None:
                    on_request_sent()

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
                    AudioMetricKey.TTFC.value: round(ttfc or 0.0, 3),
                    AudioMetricKey.END_TO_END_LATENCY.value: round(total_latency_ms, 3),
                    AudioMetricKey.CHUNK_COUNT.value: chunk_count,
                    AudioMetricKey.RAW_PCM.value: self._provider_adapter.raw_pcm,
                    AudioMetricKey.SAMPLE_RATE.value: self.config.sample_rate,
                    AudioMetricKey.INPUT_CHARS.value: len(input_text),
                    AudioMetricKey.INPUT_TOKENS.value: text_content.target_prompt_tokens
                    or 0,
                    AudioMetricKey.INPUT_TEXT.value: input_text,
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
