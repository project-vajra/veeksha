"""TTS client for streaming HTTP-based text-to-speech APIs."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, Optional

import httpx

from veeksha.client.base import BaseLLMClient
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import TTSClientConfig

logger = init_logger(__name__)

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2


def _audio_duration_ms(pcm_bytes: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Estimate audio duration from raw PCM byte count (linear16, mono)."""
    num_samples = pcm_bytes / BYTES_PER_SAMPLE
    return (num_samples / sample_rate) * 1000


class TTSClient(BaseLLMClient):
    """Async client for streaming HTTP-based TTS APIs."""

    def __init__(self, config: TTSClientConfig, **kwargs) -> None:
        self.config = config
        self.api_base = config.api_base
        self.api_key = config.api_key
        self._provider = config.provider
        self._sample_rate = config.sample_rate
        self._voice_id = config.voice_id
        self._model_id = config.model
        self._chunk_size = config.chunk_size
        self._raw_pcm = config.raw_pcm

        # Build the request URL and headers based on provider
        if self._provider == "deepgram":
            self._url = (
                f"{self.api_base}?model={self._model_id}"
                f"&encoding=linear16&container=wav&sample_rate={self._sample_rate}"
            )
            self._headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            }
        elif self._provider == "elevenlabs":
            self._url = (
                f"{self.api_base}{self._voice_id}/stream"
                f"?output_format=pcm_{self._sample_rate}"
            )
            self._headers = {
                "xi-api-key": str(self.api_key),
                "Content-Type": "application/json",
            }
        elif self._provider == "vajra":
            self._url = f"{self.api_base}/synthesize/stream"
            self._headers = {
                "Content-Type": "application/json",
            }
        elif self._provider == "voxserve":
            self._url = f"{self.api_base}/generate"
            self._headers = {}  # form-encoded, no extra headers needed
        elif self._provider == "vllm_omni":
            self._url = f"{self.api_base}/v1/audio/speech"
            self._headers = {"Content-Type": "application/json"}
        else:
            raise ValueError(f"Unsupported TTS provider: {self._provider}")

        self._client = httpx.AsyncClient(timeout=config.request_timeout)

    def _build_payload(self, text: str) -> dict:
        if self._provider in ("deepgram", "vajra"):
            return {"text": text}
        elif self._provider == "elevenlabs":
            return {"text": text, "model_id": self._model_id}
        elif self._provider == "voxserve":
            data = {"text": text, "streaming": "true"}
            if self._voice_id:
                data["speaker"] = self._voice_id
            return data
        elif self._provider == "vllm_omni":
            payload = {
                "model": self._model_id,
                "input": text,
                "stream": True,
                "response_format": "pcm",
            }
            if self._voice_id:
                payload["voice"] = self._voice_id
            return payload
        raise ValueError(f"Unsupported provider: {self._provider}")

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
    ) -> RequestResult:
        """Send a streaming TTS request and collect audio metrics."""

        # Extract text from the TEXT channel
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

        payload = self._build_payload(input_text)

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
        ttfa: Optional[float] = None
        chunk_count = 0
        total_bytes = 0
        audio_chunks: List[bytes] = []

        t_start = time.monotonic()

        try:
            # VoxServe uses form-encoded data; other providers use JSON.
            stream_kwargs: dict = {
                "headers": self._headers,
                "timeout": self.config.request_timeout,
            }
            if self._provider == "voxserve":
                stream_kwargs["data"] = payload
            else:
                stream_kwargs["json"] = payload

            async with self._client.stream(
                "POST",
                self._url,
                **stream_kwargs,
            ) as response:
                response.raise_for_status()

                async for chunk in response.aiter_bytes(chunk_size=self._chunk_size):
                    if chunk:
                        receive_time = time.monotonic()
                        if ttfa is None:
                            ttfa = (receive_time - t_start) * 1000  # ms

                        audio_chunks.append(chunk)
                        chunk_count += 1
                        total_bytes += len(chunk)

        except httpx.HTTPStatusError as e:
            error_code = e.response.status_code if e.response else 500
            error_msg = str(e)
        except httpx.ConnectError as e:
            error_code = 503
            error_msg = str(e)
        except httpx.TimeoutException:
            error_code = 408
            error_msg = "TTS request timed out"
        except Exception as e:
            error_code = 520
            error_msg = str(e)

        completed_at = time.monotonic()
        total_latency_ms = (completed_at - t_start) * 1000
        success = error_msg is None and error_code is None

        # Compute audio metrics
        if self._raw_pcm:
            pcm_bytes = total_bytes
        else:
            pcm_bytes = max(total_bytes - 44, 0)  # Deepgram WAV has 44-byte header

        audio_dur_ms = _audio_duration_ms(pcm_bytes, self._sample_rate)
        rtf = total_latency_ms / audio_dur_ms if audio_dur_ms > 0 else float("inf")

        # Combine audio bytes
        audio_data = b"".join(audio_chunks) if audio_chunks else b""

        channels = {}
        if success:
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=audio_data,
                metrics={
                    "ttft": round(ttfa or 0.0, 3),
                    "e2e": round(total_latency_ms, 3),
                    "generated_audio_duration": round(audio_dur_ms, 3),
                    "rtf": round(rtf, 5),
                    "chunk_count": chunk_count,
                    "pcm_byte_count": pcm_bytes,
                    "input_tokens": text_content.target_prompt_tokens or 0,
                    "raw_pcm": self._raw_pcm,
                    "sample_rate": self._sample_rate,
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
