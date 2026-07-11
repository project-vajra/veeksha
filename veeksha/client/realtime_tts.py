"""Websocket realtime (input-streaming) TTS client.

Speaks an OpenAI-Realtime-style protocol over a websocket: it streams text
deltas into a TTS server at an emulated upstream-LLM decode rate, receives audio
chunks back, and records per-event millisecond timestamps that the audio
interactivity evaluator consumes.

Structurally mirrors the streaming HTTP :class:`~veeksha.client.tts.TTSClient`
(adapter + client split, same metric contract), but the transport is a
websocket and the request is a paced sequence of ``input_text_buffer.append``
events rather than a single POST body.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from websockets.asyncio.client import connect
from websockets.exceptions import (
    ConnectionClosedError,
    InvalidHandshake,
    InvalidStatus,
)

from veeksha.client.base import BaseLLMClient
from veeksha.client.tts_adapters import _build_realtime_adapter
from veeksha.client.utils import TextDeltaPacer, segment_text
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.core.tts_providers import RealtimeEventKind
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import RealtimeTTSClientConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Server-side error signalling
# ---------------------------------------------------------------------------


class RealtimeServerError(Exception):
    """Raised when the server sends an ``error`` event mid-stream.

    Carries the decoded error event payload so the client can surface the
    server-provided message.
    """

    def __init__(self, event: dict) -> None:
        self.event = event
        super().__init__(str(event))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _round_ms(value: Optional[float]) -> Optional[float]:
    return round(value, 3) if value is not None else None


def _extract_output_sample_rate(session: dict) -> Optional[int]:
    """Read the server's echoed output sample rate from a ``session`` object.

    Prefers the GA location ``session.audio.output.format.rate``; falls back to a
    top-level ``sample_rate`` for servers still using the older shape.
    """
    audio = session.get("audio")
    if isinstance(audio, dict):
        output = audio.get("output")
        if isinstance(output, dict):
            fmt = output.get("format")
            if isinstance(fmt, dict) and isinstance(fmt.get("rate"), int):
                return fmt["rate"]
    rate = session.get("sample_rate")
    return rate if isinstance(rate, int) else None


class RealtimeTTSClient(BaseLLMClient):
    """Async websocket client for OpenAI-Realtime-style input-streaming TTS."""

    def __init__(self, config: "RealtimeTTSClientConfig", **kwargs) -> None:
        # **kwargs swallows tokenizer_provider (matching TTSClient); realtime
        # TTS token counts come from whitespace segmentation instead.
        super().__init__(config)
        self._realtime_config = config
        self._provider = config.provider
        self._adapter = _build_realtime_adapter(config, api_key=self.api_key)

    def _connect(self):
        """Return the websocket connect context manager.

        A seam so tests can override the transport. ``max_size=None`` lifts the
        inbound-frame cap (audio deltas can be large); ``compression=None``
        keeps binary PCM uncompressed. The asyncio transport sets ``TCP_NODELAY``
        by default, so no explicit socket option is required.
        """
        open_timeout = min(self._realtime_config.request_timeout, 30)
        return connect(
            self._adapter.build_ws_url(str(self.api_base)),
            max_size=None,
            compression=None,
            open_timeout=open_timeout,
            additional_headers=self._adapter.headers(),
        )

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent: Optional[Callable[[], None]] = None,
        on_request_dispatched: Optional[Callable[[], None]] = None,
    ) -> RequestResult:
        """Stream text deltas over a websocket and collect audio metrics."""
        text_content = request.channels.get(ChannelModality.TEXT)
        if not isinstance(text_content, TextChannelRequestContent):
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=400,
                error_msg="No TEXT channel in request for realtime TTS",
                client_completed_at=time.monotonic(),
            )

        input_text = text_content.input_text
        pacing = self._realtime_config.pacing
        segments = segment_text(input_text, pacing.tokens_per_delta)
        pacer = TextDeltaPacer(pacing, seed=pacing.seed + request.id)
        input_tokens = text_content.target_prompt_tokens or sum(
            seg.n_tokens for seg in segments
        )

        logger.debug(
            "[RealtimeTTS %s] request_id=%d session_id=%d chars=%d deltas=%d",
            self._provider,
            request.id,
            session_id,
            len(input_text),
            len(segments),
        )

        # Collected state (shared with the nested send/recv loops via closure).
        audio_chunks: list[bytes] = []
        audio_chunk_ts: list[list[float]] = []  # [[offset_ms, n_bytes], ...]
        text_delta_ts: list[list[float]] = []  # [[offset_ms, n_chars], ...]
        ttfc: Optional[float] = None
        ws_connect_latency: Optional[float] = None
        session_ready_offset: Optional[float] = None
        response_created_offset: Optional[float] = None
        input_commit_offset: Optional[float] = None
        audio_done_offset: Optional[float] = None
        response_done_offset: Optional[float] = None
        sample_rate = self._realtime_config.sample_rate

        sent_fired = False

        def fire_sent_once() -> None:
            nonlocal sent_fired
            if not sent_fired and on_request_sent is not None:
                on_request_sent()
                sent_fired = True

        error_code: Optional[int] = None
        error_msg: Optional[str] = None

        t_start = time.monotonic()

        async def send_loop(ws) -> None:
            nonlocal input_commit_offset
            if pacer.initial_delay_s > 0:
                await asyncio.sleep(pacer.initial_delay_s)
            # Pace by absolute deadlines so ws.send backpressure never
            # accumulates drift into subsequent gaps.
            deadline = time.monotonic()
            for seg in segments:
                deadline += pacer.next_gap()
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                await ws.send(self._adapter.input_append_json(seg.text))
                text_delta_ts.append([(time.monotonic() - t_start) * 1000, seg.n_chars])
            await ws.send(self._adapter.input_commit_json())
            input_commit_offset = (time.monotonic() - t_start) * 1000
            create_frame = self._adapter.response_create_json()
            if create_frame is not None:
                await ws.send(create_frame)

        async def recv_loop(ws) -> None:
            nonlocal ttfc, session_ready_offset, response_created_offset
            nonlocal audio_done_offset, response_done_offset, sample_rate
            while True:
                raw = await ws.recv()
                # Stamp receipt BEFORE any json/base64 decode work.
                offset_ms = (time.monotonic() - t_start) * 1000
                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue
                kind = self._adapter.classify(event.get("type", ""))

                if kind is RealtimeEventKind.AUDIO_DELTA:
                    chunk = self._adapter.extract_audio(event)
                    if chunk:
                        audio_chunks.append(chunk)
                        audio_chunk_ts.append([offset_ms, len(chunk)])
                        if ttfc is None:
                            ttfc = offset_ms
                        fire_sent_once()
                elif kind is RealtimeEventKind.SESSION_UPDATED:
                    if session_ready_offset is None:
                        session_ready_offset = offset_ms
                    session = event.get("session")
                    if isinstance(session, dict):
                        server_sr = _extract_output_sample_rate(session)
                        if (
                            isinstance(server_sr, int)
                            and server_sr > 0
                            and server_sr != sample_rate
                        ):
                            logger.warning(
                                "Realtime server echoed sample_rate=%d (configured "
                                "%d); using server value.",
                                server_sr,
                                self._realtime_config.sample_rate,
                            )
                            sample_rate = server_sr
                elif kind is RealtimeEventKind.RESPONSE_CREATED:
                    if response_created_offset is None:
                        response_created_offset = offset_ms
                elif kind is RealtimeEventKind.AUDIO_DONE:
                    if audio_done_offset is None:
                        audio_done_offset = offset_ms
                elif kind is RealtimeEventKind.RESPONSE_DONE:
                    response_done_offset = offset_ms
                    return  # Terminal: stop on response.done, not audio.done.
                elif kind is RealtimeEventKind.ERROR:
                    raise RealtimeServerError(event)
                # OTHER: keep receiving until response.done.

        try:
            async with asyncio.timeout(self._realtime_config.request_timeout):
                async with self._connect() as ws:
                    ws_connect_latency = (time.monotonic() - t_start) * 1000
                    await ws.send(self._adapter.session_update_json())
                    # Analog of the HTTP-200 ack: the scheduler's dispatch pacing
                    # advances on this callback.
                    if on_request_dispatched is not None:
                        on_request_dispatched()

                    async with asyncio.TaskGroup() as task_group:
                        task_group.create_task(send_loop(ws))
                        task_group.create_task(recv_loop(ws))
        except TimeoutError:
            error_code = 408
            error_msg = "Realtime TTS request timed out"
            logger.warning("Realtime TTS timeout: (%s) %s", error_code, error_msg)
        except Exception as exc:  # noqa: BLE001 - mapped to error codes below.
            error_code, error_msg = _map_error(_flatten_exception(exc))
            logger.warning("Realtime TTS error: (%s) %s", error_code, error_msg)

        completed_at = time.monotonic()
        total_latency_ms = (completed_at - t_start) * 1000
        success = error_code is None and error_msg is None

        # on_request_sent fallback: fire exactly once even on error paths.
        # With ordering="prefill" the client_runner's DispatchTracker only
        # advances via on_request_sent (or when send_request RAISES); a request
        # that fails *before* first audio would otherwise deadlock the tracker,
        # since we return an error result rather than raising. Firing here keeps
        # the sequential launch moving.
        fire_sent_once()

        audio_data = b"".join(audio_chunks) if audio_chunks else b""

        metrics = {
            AudioMetricKey.TTFC.value: round(ttfc or 0.0, 3),
            AudioMetricKey.END_TO_END_LATENCY.value: round(total_latency_ms, 3),
            AudioMetricKey.CHUNK_COUNT.value: len(audio_chunks),
            AudioMetricKey.RAW_PCM.value: self._adapter.raw_pcm,
            AudioMetricKey.SAMPLE_RATE.value: sample_rate,
            AudioMetricKey.INPUT_CHARS.value: len(input_text),
            AudioMetricKey.INPUT_TOKENS.value: input_tokens,
            AudioMetricKey.INPUT_TEXT.value: input_text,
            AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value: text_delta_ts,
            AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value: audio_chunk_ts,
            AudioMetricKey.WS_CONNECT_LATENCY_MS.value: _round_ms(ws_connect_latency),
            AudioMetricKey.SESSION_READY_OFFSET_MS.value: _round_ms(
                session_ready_offset
            ),
            AudioMetricKey.RESPONSE_CREATED_OFFSET_MS.value: _round_ms(
                response_created_offset
            ),
            AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value: _round_ms(input_commit_offset),
            AudioMetricKey.AUDIO_DONE_OFFSET_MS.value: _round_ms(audio_done_offset),
            AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value: _round_ms(
                response_done_offset
            ),
        }

        channels: dict = {}
        has_partial = bool(audio_chunks or text_delta_ts)
        if success or has_partial:
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=audio_data,
                metrics=metrics,
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


# ---------------------------------------------------------------------------
# Error flattening / mapping
# ---------------------------------------------------------------------------


def _flatten_exception(exc: BaseException) -> BaseException:
    """Reduce a (possibly nested) ExceptionGroup to its most specific leaf.

    ``asyncio.TaskGroup`` raises an ``ExceptionGroup`` bundling the failing
    task exceptions; connect-phase failures propagate bare. Collect the leaves
    and pick by priority so the mapping below sees the exception that actually
    determines the error code.
    """
    if not isinstance(exc, BaseExceptionGroup):
        return exc

    leaves: list[BaseException] = []

    def _collect(node: BaseException) -> None:
        if isinstance(node, BaseExceptionGroup):
            for sub in node.exceptions:
                _collect(sub)
        else:
            leaves.append(node)

    _collect(exc)
    if not leaves:
        return exc

    priority = (
        RealtimeServerError,
        InvalidStatus,
        InvalidHandshake,
        ConnectionClosedError,
        TimeoutError,
        OSError,
    )
    for exc_type in priority:
        for leaf in leaves:
            if isinstance(leaf, exc_type):
                return leaf
    return leaves[0]


def _map_error(exc: BaseException) -> tuple[int, str]:
    """Map a flattened exception to an (error_code, error_msg) pair."""
    if isinstance(exc, RealtimeServerError):
        event = exc.event
        message: Optional[str] = None
        if isinstance(event, dict):
            err = event.get("error")
            if isinstance(err, dict):
                message = err.get("message")
        if not message:
            message = json.dumps(event)
        return 500, message
    if isinstance(exc, InvalidStatus):
        # Handshake rejected: surface the server's HTTP status code.
        return exc.response.status_code, str(exc)
    if isinstance(exc, TimeoutError):
        return 408, "Realtime TTS request timed out"
    if isinstance(exc, (InvalidHandshake, OSError)):
        # Connect-phase failure (DNS/refused/handshake): unreachable server.
        return 503, str(exc)
    # ConnectionClosedError and anything else: unclassified transport failure.
    return 520, str(exc)
