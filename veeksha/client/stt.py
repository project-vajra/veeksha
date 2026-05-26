"""STT clients for realtime streaming speech-to-text (vajra, vllm_realtime).

`STTClient` is a thin base owning everything provider-independent: reading the
AUDIO channel, 1x pacing, error mapping, RTF, and assembling the metrics dict
the `AudioPerformanceEvaluator` consumes. Each provider is a subclass
implementing `_setup_urls` + `_transcribe`.

Both supported providers stream PCM16 over a WebSocket, paced at 1x playback
when `ws_realtime_pacing` is on. Instantiating ``STTClient(config)`` dispatches
to the subclass for ``config.provider`` (see ``__new__``), so the registry only
needs one entry.
"""

from __future__ import annotations

import asyncio
import base64
import json as json_mod
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
from veeksha.types import ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import STTClientConfig

logger = init_logger(__name__)

BYTES_PER_SAMPLE = 2

# Voxtral streaming tokens injected by the model that should be stripped
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


class STTClient(BaseLLMClient):
    """Base async client for realtime streaming STT.

    Subclasses set ``provider_name`` and implement ``_setup_urls`` +
    ``_transcribe``; the request lifecycle, pacing, error handling, and metrics
    are shared here.
    """

    # provider name -> concrete subclass, populated by __init_subclass__
    _PROVIDER_REGISTRY: dict[str, type["STTClient"]] = {}

    #: provider key matched against ``config.provider`` (set by subclasses)
    provider_name: str = ""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.provider_name:
            STTClient._PROVIDER_REGISTRY[cls.provider_name] = cls

    def __new__(cls, config: "STTClientConfig", **kwargs) -> "STTClient":
        # Dispatch STTClient(config) to the subclass for config.provider.
        target = cls
        if target is STTClient:
            try:
                target = STTClient._PROVIDER_REGISTRY[config.provider]
            except KeyError:
                raise ValueError(f"Unsupported STT provider: {config.provider}")
        return super().__new__(target)  # type: ignore[arg-type]

    def __init__(self, config: "STTClientConfig", **kwargs) -> None:
        super().__init__(config)
        self._provider = config.provider
        self._model = config.model
        self._sample_rate = config.sample_rate
        self._ws_chunk_size = config.ws_chunk_size
        self._pacing = config.ws_realtime_pacing
        self._setup_urls()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _setup_urls(self) -> None:
        """Set ``self._ws_url`` from ``self.api_base``."""

    @abstractmethod
    async def _transcribe(
        self,
        audio_path: str,
        t_start: float,
        transcript_chunks: list[str],
    ) -> tuple[Optional[float], int, dict]:
        """Stream audio to the provider and collect the transcript.

        Appends transcript text to ``transcript_chunks`` and returns
        ``(ttfc_ms, chunk_count, stream_timings)``.
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

        Simulates live-microphone cadence (1x playback) so the server sees
        audio arrive at real time. PCM16 mono: ``n_bytes / 2 / sample_rate`` s.
        """
        if self._pacing:
            await asyncio.sleep(n_bytes / 2 / self._sample_rate)

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
        stream_timings: dict = {}

        t_start = time.monotonic()

        try:
            ttfc, chunk_count, stream_timings = await self._transcribe(
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
                "ttfc": round(ttfc or 0.0, 3),
                "end_to_end_latency": round(total_latency_ms, 3),
                "generated_audio_duration": round(input_audio_duration_ms, 3),
                "rtf": round(rtf, 5),
                "chunk_count": chunk_count,
                "pcm_byte_count": file_size,
                "input_tokens": len(full_transcript.split()),
                "raw_pcm": False,
                "sample_rate": self._sample_rate,
                "transcript": full_transcript,
            }

            # Streaming-latency timings (vajra path emits these). Measured from
            # the audio stream itself, so they isolate server responsiveness
            # rather than tracking clip length the way ttfc/end_to_end_latency
            # do once 1x pacing is on.
            final_latency = stream_timings.get("final_latency_ms")
            if final_latency is not None:
                metrics_dict["final_latency"] = round(final_latency, 3)
            first_partial = stream_timings.get("first_partial_ms")
            if first_partial is not None:
                metrics_dict["first_partial"] = round(first_partial, 3)

            # Forward reference transcript for optional WER evaluation
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


class VajraSTTClient(STTClient):
    """vajra: WebSocket /stream (binary int16 PCM frames, JSON delta/done)."""

    provider_name = "vajra"

    def _setup_urls(self) -> None:
        self._ws_url = self._http_to_ws("/stream")

    async def _transcribe(
        self,
        audio_path: str,
        t_start: float,
        transcript_chunks: list[str],
    ) -> tuple[Optional[float], int, dict]:
        """Stream int16 PCM over WebSocket /stream, receiving delta/done JSON.

        Send and receive run concurrently so partial transcripts are observed
        live, rather than buffered until the upload finishes. This makes the
        returned streaming-latency timings faithful:
          - ``first_partial_ms``: first audio frame sent -> first partial
            transcript (TTFB; with 1x pacing this includes the audio that must
            be spoken before a transcript can exist).
          - ``final_latency_ms``: end-of-audio sentinel -> final transcript.
            Isolates server processing tail, independent of clip length.
        The legacy ``ttfc`` (request start -> first partial) is also returned.

        Protocol (see vajra-next/examples/asr_streaming_server.py):
          Client -> Server:  binary int16 LE PCM frames @ 16 kHz mono, then
                             text frame "end" to signal EOF.
          Server -> Client:  {"type": "ready"}
                             {"type": "delta", "text": "..."}
                             {"type": "done",  "text": "<full>"}
                             {"type": "error", "message": "..."}
        """
        import websockets

        ttfc: Optional[float] = None
        chunk_count = 0
        timings: dict = {"first_partial_ms": None, "final_latency_ms": None}
        send_marks: dict = {"audio_start": None, "audio_end": None}

        pcm_bytes = _audio_to_pcm16_bytes(audio_path, self._sample_rate)

        async with websockets.connect(self._ws_url) as ws:
            # 1. Wait for ready
            msg = json_mod.loads(await ws.recv())
            if msg.get("type") != "ready":
                raise RuntimeError(f"Expected ready, got: {msg}")

            # 2. Sender coroutine: stream PCM16 frames, pace at 1x when on, then
            #    the EOF sentinel. Runs concurrently with the receive loop so
            #    deltas emitted mid-stream are timed when they actually arrive.
            async def _send() -> None:
                send_marks["audio_start"] = time.monotonic()
                for i in range(0, len(pcm_bytes), self._ws_chunk_size):
                    chunk = pcm_bytes[i : i + self._ws_chunk_size]
                    await ws.send(chunk)
                    await self._maybe_pace(len(chunk))
                await ws.send("end")
                send_marks["audio_end"] = time.monotonic()

            send_task = asyncio.ensure_future(_send())
            try:
                # 3. Consume deltas + done concurrently with the sender.
                while True:
                    msg = json_mod.loads(await ws.recv())
                    msg_type = msg.get("type")

                    if msg_type == "delta":
                        now = time.monotonic()
                        if ttfc is None:
                            ttfc = (now - t_start) * 1000
                            if send_marks["audio_start"] is not None:
                                timings["first_partial_ms"] = (
                                    now - send_marks["audio_start"]
                                ) * 1000
                        transcript_chunks.append(msg.get("text", ""))
                        chunk_count += 1

                    elif msg_type == "done":
                        now = time.monotonic()
                        if not transcript_chunks and msg.get("text"):
                            transcript_chunks.append(msg["text"])
                            chunk_count = 1
                            if ttfc is None:
                                ttfc = (now - t_start) * 1000
                                if send_marks["audio_start"] is not None:
                                    timings["first_partial_ms"] = (
                                        now - send_marks["audio_start"]
                                    ) * 1000
                        if send_marks["audio_end"] is not None:
                            timings["final_latency_ms"] = (
                                now - send_marks["audio_end"]
                            ) * 1000
                        break

                    elif msg_type == "error":
                        raise RuntimeError(
                            f"Vajra streaming error: {msg.get('message', 'unknown')}"
                        )
            finally:
                # On the normal path the sender is already done (server only
                # emits "done" after "end"); on the error path, cancel it. A
                # CancelledError here is expected; any other is a real send
                # failure and must propagate.
                if not send_task.done():
                    send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

        return ttfc, chunk_count, timings


class VllmRealtimeSTTClient(STTClient):
    """vllm_realtime: WebSocket /v1/realtime (base64 PCM16 chunks, deltas)."""

    provider_name = "vllm_realtime"

    def _setup_urls(self) -> None:
        self._ws_url = self._http_to_ws("/v1/realtime")

    async def _transcribe(
        self,
        audio_path: str,
        t_start: float,
        transcript_chunks: list[str],
    ) -> tuple[Optional[float], int, dict]:
        """Stream PCM16 over WebSocket /v1/realtime, receiving transcription deltas."""
        import websockets

        ttfc: Optional[float] = None
        chunk_count = 0

        async with websockets.connect(self._ws_url) as ws:
            # 1. Wait for session.created
            msg = json_mod.loads(await ws.recv())
            if msg.get("type") != "session.created":
                raise RuntimeError(f"Expected session.created, got: {msg}")

            # 2. Configure the session with the model
            await ws.send(json_mod.dumps({
                "type": "session.update",
                "model": self._model,
            }))
            await ws.send(json_mod.dumps({
                "type": "input_audio_buffer.commit",
            }))

            # 3. Convert audio to PCM16 @ target sample rate and send chunks,
            #    paced at 1x when ws_realtime_pacing is on.
            pcm_bytes = _audio_to_pcm16_bytes(audio_path, self._sample_rate)
            for i in range(0, len(pcm_bytes), self._ws_chunk_size):
                chunk = pcm_bytes[i : i + self._ws_chunk_size]
                await ws.send(json_mod.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                }))
                await self._maybe_pace(len(chunk))

            # 4. Signal all audio sent
            await ws.send(json_mod.dumps({
                "type": "input_audio_buffer.commit",
                "final": True,
            }))

            # 5. Receive transcription deltas until done
            while True:
                msg = json_mod.loads(await ws.recv())
                msg_type = msg.get("type")

                if msg_type == "transcription.delta":
                    if ttfc is None:
                        ttfc = (time.monotonic() - t_start) * 1000
                    transcript_chunks.append(msg["delta"])
                    chunk_count += 1

                elif msg_type == "transcription.done":
                    # Use the final text if we got no deltas
                    if not transcript_chunks and msg.get("text"):
                        transcript_chunks.append(msg["text"])
                        chunk_count = 1
                        if ttfc is None:
                            ttfc = (time.monotonic() - t_start) * 1000
                    break

                elif msg_type == "error":
                    raise RuntimeError(
                        f"Realtime STT error: {msg.get('error', 'unknown')}"
                    )

        return ttfc, chunk_count, {}
