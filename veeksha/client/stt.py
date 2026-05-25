"""STT client for speech-to-text APIs (vajra, vllm, vllm_realtime)."""

from __future__ import annotations

import asyncio
import base64
import json as json_mod
import os
import re
import time
from typing import TYPE_CHECKING, Optional

import httpx
import librosa
import numpy as np

from veeksha.client.base import BaseLLMClient
from veeksha.core.request import Request
from veeksha.core.request_content import AudioChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import STTClientConfig

logger = init_logger(__name__)

BYTES_PER_SAMPLE = 2

# Voxtral streaming tokens injected by the model that should be stripped
_STREAMING_TOKEN_RE = re.compile(r"\[STREAMING_(?:PAD|WORD)\]")


def _audio_duration_from_file(file_path: str, sample_rate: int) -> float:
    """Estimate audio duration in ms from WAV file size (linear16 mono).

    Falls back to file-size heuristic for non-WAV formats.
    """
    try:
        file_size = os.path.getsize(file_path)
    except OSError:
        return 0.0
    pcm_bytes = max(file_size - 44, 0)
    num_samples = pcm_bytes / BYTES_PER_SAMPLE
    return (num_samples / sample_rate) * 1000


def _clean_transcript(text: str) -> str:
    """Clean up a streaming transcript.

    Strips Voxtral streaming control tokens and collapses whitespace.
    """
    text = _STREAMING_TOKEN_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _audio_to_pcm16_bytes(audio_path: str, target_sr: int = 16000) -> bytes:
    """Load an audio file and convert to raw PCM16 bytes at target sample rate."""
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


class STTClient(BaseLLMClient):
    """Async client for speech-to-text APIs.

    Supported providers:
      - vajra:         POST /transcribe  (multipart, batch)
                       OR WebSocket /stream when streaming=true
      - vllm:          POST /v1/audio/transcriptions  (multipart, batch or SSE)
      - vllm_realtime: WebSocket /v1/realtime  (PCM16 chunks, streaming deltas)
    """

    def __init__(self, config: STTClientConfig, **kwargs) -> None:
        self.config = config
        self.api_base = config.api_base
        self.api_key = config.api_key
        self._provider = config.provider
        self._model = config.model
        self._language = config.language
        self._audio_format = config.audio_format
        self._sample_rate = config.sample_rate
        self._streaming = config.streaming
        self._mime_type = config.get_mime_type()
        self._ws_chunk_size = config.ws_chunk_size

        if self._provider == "vajra":
            self._url = f"{self.api_base}/transcribe"
            if self._streaming:
                api_base = self.api_base or ""
                if api_base.startswith("https://"):
                    ws_base = "wss://" + api_base[len("https://"):]
                elif api_base.startswith("http://"):
                    ws_base = "ws://" + api_base[len("http://"):]
                else:
                    ws_base = api_base
                self._ws_url = f"{ws_base}/stream"
        elif self._provider == "vllm":
            self._url = f"{self.api_base}/v1/audio/transcriptions"
        elif self._provider == "vllm_realtime":
            # Convert http(s) to ws(s) for WebSocket
            api_base = self.api_base or ""
            if api_base.startswith("https://"):
                ws_base = "wss://" + api_base[len("https://"):]
            elif api_base.startswith("http://"):
                ws_base = "ws://" + api_base[len("http://"):]
            else:
                ws_base = api_base
            self._ws_url = f"{ws_base}/v1/realtime"
        else:
            raise ValueError(f"Unsupported STT provider: {self._provider}")

        self._headers: dict[str, str] = {}
        if self.api_key and self._provider in ("vllm", "vllm_realtime"):
            self._headers["Authorization"] = f"Bearer {self.api_key}"

        # HTTP client not needed for pure-WebSocket providers
        if not (self._provider == "vllm_realtime"
                or (self._provider == "vajra" and self._streaming)):
            self._client = httpx.AsyncClient(timeout=config.request_timeout)

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
    ) -> RequestResult:
        """Send an audio file to the STT API and collect transcription metrics."""

        audio_content = request.channels.get(ChannelModality.AUDIO)
        if not isinstance(audio_content, AudioChannelRequestContent):
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=400,
                error_msg="No AUDIO channel in request for STT",
                client_completed_at=time.monotonic(),
            )

        audio_path = audio_content.input_audio

        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()
        except (OSError, IOError) as e:
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=404,
                error_msg=f"Failed to read audio file {audio_path}: {e}",
                client_completed_at=time.monotonic(),
            )

        input_audio_duration_ms = _audio_duration_from_file(
            audio_path, self._sample_rate
        )

        logger.debug(
            "[STT %s] request_id=%d session_id=%d file=%s duration_ms=%.1f",
            self._provider,
            request.id,
            session_id,
            audio_path,
            input_audio_duration_ms,
        )

        error_msg: Optional[str] = None
        error_code: Optional[int] = None
        ttft: Optional[float] = None
        tpot: Optional[float] = None
        transcript_chunks: list[str] = []
        chunk_count = 0

        t_start = time.monotonic()

        try:
            if self._provider == "vajra":
                if self._streaming:
                    ttft, chunk_count = await self._stream_vajra(
                        audio_path, t_start, transcript_chunks
                    )
                else:
                    text, server_ttft, server_tpot = await self._batch_vajra(audio_data, audio_path)
                    ttft = server_ttft if server_ttft is not None else (time.monotonic() - t_start) * 1000
                    tpot = server_tpot
                    transcript_chunks.append(text)
                    chunk_count = 1

            elif self._provider == "vllm":
                if self._streaming:
                    ttft, chunk_count = await self._stream_vllm(
                        audio_data, audio_path, t_start, transcript_chunks
                    )
                else:
                    text = await self._batch_vllm(audio_data, audio_path)
                    ttft = (time.monotonic() - t_start) * 1000
                    transcript_chunks.append(text)
                    chunk_count = 1

            elif self._provider == "vllm_realtime":
                ttft, chunk_count = await self._realtime_vllm(
                    audio_path, t_start, transcript_chunks
                )

        except httpx.HTTPStatusError as e:
            error_code = e.response.status_code if e.response else 500
            error_msg = str(e)
        except httpx.ConnectError as e:
            error_code = 503
            error_msg = str(e)
        except httpx.TimeoutException:
            error_code = 408
            error_msg = "STT request timed out"
        except Exception as e:
            error_code = 520
            error_msg = str(e)
            logger.error("[STT %s] request_id=%d error: %s", self._provider, request.id, e, exc_info=True)

        completed_at = time.monotonic()
        total_latency_ms = (completed_at - t_start) * 1000
        success = error_msg is None and error_code is None

        rtf = (
            total_latency_ms / input_audio_duration_ms
            if input_audio_duration_ms > 0
            else float("inf")
        )

        full_transcript = _clean_transcript("".join(transcript_chunks))

        channels = {}
        if success:
            # Return metrics via the AUDIO channel so that
            # AudioPerformanceEvaluator can aggregate them unchanged.
            metrics_dict: dict = {
                "ttft": round(ttft or 0.0, 3),
                "e2e": round(total_latency_ms, 3),
                "generated_audio_duration": round(input_audio_duration_ms, 3),
                "rtf": round(rtf, 5),
                "chunk_count": chunk_count,
                "pcm_byte_count": len(audio_data),
                "input_tokens": len(full_transcript.split()),
                "raw_pcm": False,
                "sample_rate": self._sample_rate,
                "transcript": full_transcript,
            }
            if tpot is not None:
                metrics_dict["tpot"] = round(tpot, 3)

            # Forward reference transcript for WER evaluation
            expected = request.metadata.get("expected_transcript")
            if expected is not None:
                metrics_dict["expected_transcript"] = expected

            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=full_transcript,
                metrics=metrics_dict,
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

    # ------------------------------------------------------------------
    # Vajra — POST /transcribe  (multipart, batch)
    # ------------------------------------------------------------------

    async def _batch_vajra(
        self, audio_data: bytes, audio_path: str,
    ) -> tuple[str, Optional[float], Optional[float]]:
        """Returns (text, ttft_ms, tpot_ms)."""
        filename = os.path.basename(audio_path)
        files = {"audio": (filename, audio_data, self._mime_type)}
        response = await self._client.post(
            self._url,
            files=files,
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        server_ttft = data.get("ttft_ms")
        # Compute TPOT from server-reported decode time and token count
        decode_ms = data.get("decode_ms")
        num_tokens = data.get("num_tokens")
        tpot: Optional[float] = None
        if decode_ms is not None and num_tokens and num_tokens > 0:
            tpot = decode_ms / num_tokens
        return data.get("text", ""), server_ttft, tpot

    # ------------------------------------------------------------------
    # Vajra — WebSocket /stream  (binary PCM16, JSON delta frames)
    # ------------------------------------------------------------------

    async def _stream_vajra(
        self,
        audio_path: str,
        t_start: float,
        transcript_chunks: list[str],
    ) -> tuple[Optional[float], int]:
        """Stream int16 PCM over WebSocket /stream, receive delta/done JSON.

        Protocol (see vajra-next/examples/asr_streaming_server.py):
          Client -> Server:  binary int16 LE PCM frames @ 16 kHz mono, then
                             text frame "end" to signal EOF.
          Server -> Client:  {"type": "ready"}
                             {"type": "delta", "text": "..."}
                             {"type": "done",  "text": "<full>"}
                             {"type": "error", "message": "..."}
        """
        import websockets

        ttft: Optional[float] = None
        chunk_count = 0

        async with websockets.connect(self._ws_url) as ws:
            # 1. Wait for ready
            msg = json_mod.loads(await ws.recv())
            if msg.get("type") != "ready":
                raise RuntimeError(f"Expected ready, got: {msg}")

            # 2. Send PCM16 audio as binary frames of ws_chunk_size bytes each.
            #    When ws_realtime_pacing is on, sleep chunk_duration between
            #    sends to simulate live microphone cadence (1× playback).
            pcm_bytes = _audio_to_pcm16_bytes(audio_path, self._sample_rate)
            pace = self.config.ws_realtime_pacing
            for i in range(0, len(pcm_bytes), self._ws_chunk_size):
                chunk = pcm_bytes[i : i + self._ws_chunk_size]
                await ws.send(chunk)
                if pace:
                    await asyncio.sleep(len(chunk) / 2 / self._sample_rate)

            # 3. EOF sentinel
            await ws.send("end")

            # 4. Consume deltas + done
            while True:
                msg = json_mod.loads(await ws.recv())
                msg_type = msg.get("type")

                if msg_type == "delta":
                    if ttft is None:
                        ttft = (time.monotonic() - t_start) * 1000
                    transcript_chunks.append(msg.get("text", ""))
                    chunk_count += 1

                elif msg_type == "done":
                    if not transcript_chunks and msg.get("text"):
                        transcript_chunks.append(msg["text"])
                        chunk_count = 1
                        if ttft is None:
                            ttft = (time.monotonic() - t_start) * 1000
                    break

                elif msg_type == "error":
                    raise RuntimeError(
                        f"Vajra streaming error: {msg.get('message', 'unknown')}"
                    )

        return ttft, chunk_count

    # ------------------------------------------------------------------
    # vLLM — POST /v1/audio/transcriptions  (multipart, batch or SSE)
    # ------------------------------------------------------------------

    async def _batch_vllm(self, audio_data: bytes, audio_path: str) -> str:
        filename = os.path.basename(audio_path)
        files = {"file": (filename, audio_data, self._mime_type)}
        data: dict[str, str] = {
            "model": self._model,
            "language": self._language,
            "response_format": "json",
        }
        response = await self._client.post(
            self._url,
            headers=self._headers,
            files=files,
            data=data,
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("text", "")

    async def _stream_vllm(
        self,
        audio_data: bytes,
        audio_path: str,
        t_start: float,
        transcript_chunks: list[str],
    ) -> tuple[Optional[float], int]:
        """SSE streaming via /v1/audio/transcriptions?stream=true."""
        filename = os.path.basename(audio_path)
        files = {"file": (filename, audio_data, self._mime_type)}
        data: dict[str, str] = {
            "model": self._model,
            "language": self._language,
            "response_format": "json",
            "stream": "true",
        }

        ttft: Optional[float] = None
        chunk_count = 0

        async with self._client.stream(
            "POST",
            self._url,
            headers=self._headers,
            files=files,
            data=data,
            timeout=self.config.request_timeout,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[len("data: "):]
                try:
                    event = json_mod.loads(line)
                except (json_mod.JSONDecodeError, ValueError):
                    continue

                for choice in event.get("choices", []):
                    content = choice.get("delta", {}).get("content", "")
                    if content:
                        if ttft is None:
                            ttft = (time.monotonic() - t_start) * 1000
                        transcript_chunks.append(content)
                        chunk_count += 1

        return ttft, chunk_count

    # ------------------------------------------------------------------
    # vLLM Realtime — WebSocket /v1/realtime  (PCM16 chunks)
    # ------------------------------------------------------------------

    async def _realtime_vllm(
        self,
        audio_path: str,
        t_start: float,
        transcript_chunks: list[str],
    ) -> tuple[Optional[float], int]:
        """Stream PCM16 audio over WebSocket, receive transcription deltas."""
        import websockets

        ttft: Optional[float] = None
        chunk_count = 0

        async with websockets.connect(self._ws_url) as ws:
            # 1. Wait for session.created
            msg = json_mod.loads(await ws.recv())
            if msg.get("type") != "session.created":
                raise RuntimeError(f"Expected session.created, got: {msg}")

            # 2. Send session.update with model
            await ws.send(json_mod.dumps({
                "type": "session.update",
                "model": self._model,
            }))

            # 3. Signal ready
            await ws.send(json_mod.dumps({
                "type": "input_audio_buffer.commit",
            }))

            # 4. Convert audio to PCM16 @ target sample rate and send chunks.
            #    When ws_realtime_pacing is on, sleep chunk_duration between
            #    sends to simulate live microphone cadence (1× playback).
            pcm_bytes = _audio_to_pcm16_bytes(audio_path, self._sample_rate)
            pace = self.config.ws_realtime_pacing
            for i in range(0, len(pcm_bytes), self._ws_chunk_size):
                chunk = pcm_bytes[i : i + self._ws_chunk_size]
                await ws.send(json_mod.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                }))
                if pace:
                    await asyncio.sleep(len(chunk) / 2 / self._sample_rate)

            # 5. Signal all audio sent
            await ws.send(json_mod.dumps({
                "type": "input_audio_buffer.commit",
                "final": True,
            }))

            # 6. Receive transcription deltas until done
            while True:
                msg = json_mod.loads(await ws.recv())
                msg_type = msg.get("type")

                if msg_type == "transcription.delta":
                    if ttft is None:
                        ttft = (time.monotonic() - t_start) * 1000
                    transcript_chunks.append(msg["delta"])
                    chunk_count += 1

                elif msg_type == "transcription.done":
                    # Use the final text if we got no deltas
                    if not transcript_chunks and msg.get("text"):
                        transcript_chunks.append(msg["text"])
                        chunk_count = 1
                        if ttft is None:
                            ttft = (time.monotonic() - t_start) * 1000
                    break

                elif msg_type == "error":
                    raise RuntimeError(
                        f"Realtime STT error: {msg.get('error', 'unknown')}"
                    )

        return ttft, chunk_count
