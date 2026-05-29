"""STT clients for realtime streaming speech-to-text (vajra, vllm_realtime).

Both providers stream PCM16 over a WebSocket and report transcription metrics
(ttfc, end-to-end latency, RTF). They share one lifecycle in ``_STTClientBase``:
audio is paced at 1x playback when ``ws_realtime_pacing`` is on, and the send
and receive loops run concurrently so ``ttfc`` reflects when the first partial
actually arrives rather than when the upload finishes. Each provider only
supplies four small protocol hooks (open session, encode a chunk, EOF sentinel,
parse a message).

``STTClient(config)`` is a factory returning the client for ``config.provider``
(see ``_PROVIDERS``), so the registry needs only one entry.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import librosa
import numpy as np

from veeksha.client.base import BaseLLMClient
from veeksha.core.request import Request
from veeksha.core.request_content import AudioChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import STTClientConfig

logger = init_logger(__name__)

BYTES_PER_SAMPLE = 2

# Voxtral streaming tokens injected by the model that should be stripped.
_STREAMING_TOKEN_RE = re.compile(r"\[STREAMING_(?:PAD|WORD)\]")


def _wav_duration_ms(file_size_bytes: int, sample_rate: int) -> float:
    """Estimate WAV audio duration in ms from file size (linear16 mono)."""
    pcm_bytes = max(file_size_bytes - 44, 0)
    return (pcm_bytes / BYTES_PER_SAMPLE / sample_rate) * 1000


def _clean_transcript(text: str) -> str:
    """Strip Voxtral streaming control tokens and collapse whitespace."""
    text = _STREAMING_TOKEN_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _audio_to_pcm16_bytes(audio_path: str, target_sr: int) -> bytes:
    """Load an audio file and convert to raw PCM16 bytes at target sample rate."""
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


class _STTClientBase(BaseLLMClient):
    """Shared streaming lifecycle for realtime STT providers.

    Subclasses set ``ws_path`` and implement the four protocol hooks; the
    request lifecycle, 1x pacing, concurrent stream, error mapping, and metrics
    assembly live here.
    """

    #: WebSocket path appended to ``api_base`` (set by subclasses).
    ws_path: str = ""

    def __init__(self, config: "STTClientConfig", **kwargs) -> None:
        super().__init__(config)
        self._provider = config.provider
        self._model = config.model
        self._sample_rate = config.sample_rate
        self._ws_chunk_size = config.ws_chunk_size
        self._pacing = config.ws_realtime_pacing
        self._ws_url = self._http_to_ws(self.ws_path)

    # ------------------------------------------------------------------
    # Provider protocol hooks
    # ------------------------------------------------------------------

    @abstractmethod
    async def _open_session(self, ws) -> None:
        """Complete the provider handshake before audio is sent."""

    @abstractmethod
    def _encode_chunk(self, chunk: bytes) -> str | bytes:
        """Frame a PCM16 chunk into the message the provider expects."""
        raise NotImplementedError

    @abstractmethod
    def _eof(self) -> str | bytes:
        """Return the end-of-audio sentinel message."""
        raise NotImplementedError

    @abstractmethod
    def _parse_message(self, msg: dict) -> tuple[str, str]:
        """Map a server message to ``(kind, text)``.

        ``kind`` is ``"delta"``, ``"done"``, ``"error"``, or ``""`` (ignore).
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _http_to_ws(self, path: str) -> str:
        """Convert the http(s) api_base to a ws(s) URL with ``path`` appended."""
        base = self.api_base or ""
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        return f"{base}{path}"

    async def _maybe_pace(self, n_bytes: int) -> None:
        """Sleep one chunk's playback duration when realtime pacing is on.

        Simulates live-microphone cadence (1x playback). PCM16 mono:
        ``n_bytes / 2 / sample_rate`` seconds.
        """
        if self._pacing:
            await asyncio.sleep(n_bytes / BYTES_PER_SAMPLE / self._sample_rate)

    async def _stream(
        self,
        audio_path: str,
        t_start: float,
        transcript_chunks: list[str],
    ) -> tuple[Optional[float], int]:
        """Stream PCM16 to the provider and collect the transcript.

        The sender and receiver run concurrently so partial transcripts are
        timed when they arrive, not after the (paced) upload completes. Returns
        ``(ttfc_ms, chunk_count)``.
        """
        import websockets

        ttfc: Optional[float] = None
        chunk_count = 0
        pcm_bytes = _audio_to_pcm16_bytes(audio_path, self._sample_rate)

        async with websockets.connect(self._ws_url) as ws:
            await self._open_session(ws)

            async def _send() -> None:
                for i in range(0, len(pcm_bytes), self._ws_chunk_size):
                    chunk = pcm_bytes[i : i + self._ws_chunk_size]
                    await ws.send(self._encode_chunk(chunk))
                    await self._maybe_pace(len(chunk))
                await ws.send(self._eof())

            send_task = asyncio.ensure_future(_send())
            try:
                while True:
                    kind, text = self._parse_message(json.loads(await ws.recv()))
                    if kind == "delta":
                        if ttfc is None:
                            ttfc = (time.monotonic() - t_start) * 1000
                        transcript_chunks.append(text)
                        chunk_count += 1
                    elif kind == "done":
                        # Fall back to the final text if no deltas streamed.
                        if not transcript_chunks and text:
                            transcript_chunks.append(text)
                            chunk_count = 1
                            if ttfc is None:
                                ttfc = (time.monotonic() - t_start) * 1000
                        break
                    elif kind == "error":
                        raise RuntimeError(
                            f"{self._provider} streaming error: {text or 'unknown'}"
                        )
            finally:
                # Normal path: the sender is already done (servers only emit
                # "done" after EOF). Error path: cancel it. A CancelledError is
                # expected; any other error is a real send failure and must
                # propagate.
                if not send_task.done():
                    send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

        return ttfc, chunk_count

    # ------------------------------------------------------------------
    # Request lifecycle (shared)
    # ------------------------------------------------------------------

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
    ) -> RequestResult:
        """Stream an audio file to the STT API and collect transcription metrics."""

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
            file_size = os.path.getsize(audio_path)
        except OSError as e:
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=404,
                error_msg=f"Failed to read audio file {audio_path}: {e}",
                client_completed_at=time.monotonic(),
            )

        input_audio_duration_ms = _wav_duration_ms(file_size, self._sample_rate)

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
        ttfc: Optional[float] = None
        transcript_chunks: list[str] = []
        chunk_count = 0

        t_start = time.monotonic()

        try:
            ttfc, chunk_count = await self._stream(
                audio_path, t_start, transcript_chunks
            )
        except asyncio.TimeoutError:
            error_code = 408
            error_msg = "STT request timed out"
        except (OSError, ConnectionError) as e:
            # Includes ConnectionRefusedError when the server isn't up.
            error_code = 503
            error_msg = str(e)
        except Exception as e:
            error_code = 520
            error_msg = str(e)
            logger.error(
                "[STT %s] request_id=%d error: %s",
                self._provider,
                request.id,
                e,
                exc_info=True,
            )

        completed_at = time.monotonic()
        total_latency_ms = (completed_at - t_start) * 1000
        success = error_msg is None and error_code is None

        full_transcript = _clean_transcript("".join(transcript_chunks))

        channels = {}
        if success:
            # Report the input clip's byte count; the evaluator derives
            # duration/RTF from pcm_byte_count + sample_rate.
            metrics_dict: dict = {
                "audio_task": AudioTask.STT,
                "ttfc": round(ttfc or 0.0, 3),
                "end_to_end_latency": round(total_latency_ms, 3),
                "chunk_count": chunk_count,
                "pcm_byte_count": file_size,
                "input_tokens": len(full_transcript.split()),
                "sample_rate": self._sample_rate,
                "transcript": full_transcript,
            }

            # Ground truth is guaranteed by the audio trace generator.
            metrics_dict["expected_transcript"] = request.metadata[
                "expected_transcript"
            ]

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


class VajraSTTClient(_STTClientBase):
    """vajra: WebSocket /stream — binary int16 PCM frames, JSON delta/done.

    Client -> Server: int16 LE PCM frames, then a text frame ``"end"``.
    Server -> Client: ``{"type": "ready"}`` then ``delta`` / ``done`` / ``error``.
    """

    ws_path = "/stream"

    async def _open_session(self, ws) -> None:
        msg = json.loads(await ws.recv())
        if msg.get("type") != "ready":
            raise RuntimeError(f"Expected ready, got: {msg}")

    def _encode_chunk(self, chunk: bytes) -> bytes:
        return chunk

    def _eof(self) -> str:
        return "end"

    def _parse_message(self, msg: dict) -> tuple[str, str]:
        msg_type = msg.get("type")
        if msg_type == "delta":
            return "delta", msg.get("text", "")
        if msg_type == "done":
            return "done", msg.get("text", "")
        if msg_type == "error":
            return "error", msg.get("message", "")
        return "", ""


class VllmRealtimeSTTClient(_STTClientBase):
    """vllm_realtime: WebSocket /v1/realtime — base64 PCM16 chunks, deltas.

    Server -> Client: ``{"type": "session.created"}`` then
    ``transcription.delta`` / ``transcription.done`` / ``error``.
    """

    ws_path = "/v1/realtime"

    async def _open_session(self, ws) -> None:
        msg = json.loads(await ws.recv())
        if msg.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got: {msg}")
        await ws.send(json.dumps({"type": "session.update", "model": self._model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    def _encode_chunk(self, chunk: bytes) -> str:
        return json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            }
        )

    def _eof(self) -> str:
        return json.dumps({"type": "input_audio_buffer.commit", "final": True})

    def _parse_message(self, msg: dict) -> tuple[str, str]:
        msg_type = msg.get("type")
        if msg_type == "transcription.delta":
            return "delta", msg.get("delta", "")
        if msg_type == "transcription.done":
            return "done", msg.get("text", "")
        if msg_type == "error":
            return "error", str(msg.get("error", ""))
        return "", ""


_PROVIDERS: dict[str, type[_STTClientBase]] = {
    "vajra": VajraSTTClient,
    "vllm_realtime": VllmRealtimeSTTClient,
}


def STTClient(config: "STTClientConfig", **kwargs) -> _STTClientBase:
    """Factory: return the STT client for ``config.provider``."""
    try:
        cls = _PROVIDERS[config.provider]
    except KeyError:
        raise ValueError(f"Unsupported STT provider: {config.provider}")
    return cls(config, **kwargs)
