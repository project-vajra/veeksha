"""Native cloud-provider WebSocket clients for streaming text-to-speech.

Both clients emit Veeksha's transport-independent raw PCM timing contract, so
the same first-playable, stall, RTF, duplex-overlap, and fluidity evaluators can
be applied to Vajra, ElevenLabs, and Deepgram without provider-side clocks.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.parse import quote, urlencode, urljoin

from websockets.asyncio.client import connect
from websockets.exceptions import InvalidHandshake, InvalidStatus

from veeksha.client.base import BaseLLMClient
from veeksha.client.utils import TextDeltaPacer, segment_text
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import (
        DeepgramAuraStreamingTTSClientConfig,
        DeepgramFluxStreamingTTSClientConfig,
        ElevenLabsStreamingTTSClientConfig,
    )

logger = init_logger(__name__)


@dataclass(frozen=True)
class NativeProtocolEvent:
    """One normalized provider event."""

    audio: bytes = b""
    ready: bool = False
    response_started: bool = False
    audio_done: bool = False
    terminal: bool = False
    error: Optional[str] = None


class NativeStreamingTTSError(Exception):
    """Fatal provider event received after a successful WebSocket handshake."""


class ElevenLabsStreamingProtocol:
    provider = "elevenlabs"
    protocol_name = "v1_stream_input"
    has_ready_event = False

    def __init__(
        self, config: "ElevenLabsStreamingTTSClientConfig", api_key: str
    ) -> None:
        self.config = config
        self.api_key = api_key

    def build_ws_url(self, api_base: str) -> str:
        normalized = api_base.rstrip("/") + "/"
        path = f"v1/text-to-speech/{quote(self.config.voice_id, safe='')}/stream-input"
        query = urlencode(
            {
                "model_id": self.config.model,
                "output_format": f"pcm_{self.config.sample_rate}",
                "auto_mode": str(self.config.auto_mode).lower(),
                "apply_text_normalization": self.config.apply_text_normalization,
            }
        )
        url = urljoin(normalized, path)
        if url.startswith("https://"):
            url = "wss://" + url[len("https://") :]
        elif url.startswith("http://"):
            url = "ws://" + url[len("http://") :]
        return f"{url}?{query}"

    def headers(self) -> dict[str, str]:
        return {"xi-api-key": self.api_key}

    def initial_messages(self) -> list[str]:
        payload: dict[str, Any] = {
            "text": " ",
            "voice_settings": {
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost,
                "speed": self.config.speed,
            },
        }
        if not self.config.auto_mode:
            payload["generation_config"] = {
                "chunk_length_schedule": self.config.chunk_length_schedule
            }
        return [json.dumps(payload)]

    def text_message(self, text: str) -> str:
        return json.dumps({"text": text})

    def finish_message(self) -> str:
        # An empty text message flushes buffered text and closes the generation.
        return json.dumps({"text": ""})

    def parse(self, raw: str | bytes) -> NativeProtocolEvent:
        if isinstance(raw, bytes):
            return NativeProtocolEvent(error="Unexpected binary ElevenLabs frame")
        try:
            event = json.loads(raw)
        except json.JSONDecodeError, TypeError, ValueError:
            return NativeProtocolEvent()
        if not isinstance(event, dict):
            return NativeProtocolEvent()
        error = event.get("error")
        if error:
            if isinstance(error, dict):
                message = error.get("message") or json.dumps(error)
            else:
                message = str(error)
            return NativeProtocolEvent(error=message)
        audio = b""
        encoded_audio = event.get("audio")
        if isinstance(encoded_audio, str) and encoded_audio:
            try:
                audio = base64.b64decode(encoded_audio, validate=True)
            except (ValueError, TypeError) as exc:
                return NativeProtocolEvent(error=f"Invalid ElevenLabs audio: {exc}")
        terminal = bool(event.get("isFinal"))
        return NativeProtocolEvent(
            audio=audio,
            response_started=bool(audio),
            audio_done=terminal,
            terminal=terminal,
        )


class DeepgramFluxStreamingProtocol:
    provider = "deepgram"
    protocol_name = "v2_flux_speak"
    has_ready_event = True

    def __init__(
        self, config: "DeepgramFluxStreamingTTSClientConfig", api_key: str
    ) -> None:
        self.config = config
        self.api_key = api_key

    def build_ws_url(self, api_base: str) -> str:
        normalized = api_base.rstrip("/") + "/"
        query = urlencode(
            {
                "model": self.config.model,
                "encoding": "linear16",
                "sample_rate": self.config.sample_rate,
                "mip_opt_out": str(self.config.mip_opt_out).lower(),
            }
        )
        url = urljoin(normalized, "v2/speak")
        if url.startswith("https://"):
            url = "wss://" + url[len("https://") :]
        elif url.startswith("http://"):
            url = "ws://" + url[len("http://") :]
        return f"{url}?{query}"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key}"}

    def initial_messages(self) -> list[str]:
        return []

    def text_message(self, text: str) -> str:
        return json.dumps({"type": "Speak", "text": text})

    def finish_message(self) -> str:
        return json.dumps({"type": "Flush"})

    def parse(self, raw: str | bytes) -> NativeProtocolEvent:
        if isinstance(raw, bytes):
            return NativeProtocolEvent(audio=raw, response_started=bool(raw))
        try:
            event = json.loads(raw)
        except json.JSONDecodeError, TypeError, ValueError:
            return NativeProtocolEvent()
        if not isinstance(event, dict):
            return NativeProtocolEvent()
        event_type = event.get("type")
        if event_type == "Error":
            return NativeProtocolEvent(
                error=str(event.get("description") or event.get("code") or event)
            )
        return NativeProtocolEvent(
            ready=event_type == "Connected",
            response_started=event_type == "SpeechStarted",
            audio_done=event_type == "SpeechMetadata",
            terminal=event_type == "SpeechMetadata",
        )


class DeepgramAuraStreamingProtocol:
    """Deepgram Aura continuous-text WebSocket protocol.

    The shared client sends text incrementally and records whether Aura emits
    audio before the final ``Flush`` instead of assuming a particular buffering
    policy.
    """

    provider = "deepgram"
    protocol_name = "v1_aura_speak"
    has_ready_event = False

    def __init__(
        self, config: "DeepgramAuraStreamingTTSClientConfig", api_key: str
    ) -> None:
        self.config = config
        self.api_key = api_key

    def build_ws_url(self, api_base: str) -> str:
        normalized = api_base.rstrip("/") + "/"
        query = urlencode(
            {
                "model": self.config.model,
                "encoding": "linear16",
                "sample_rate": self.config.sample_rate,
                "mip_opt_out": str(self.config.mip_opt_out).lower(),
                "speed": self.config.speed,
            }
        )
        url = urljoin(normalized, "v1/speak")
        if url.startswith("https://"):
            url = "wss://" + url[len("https://") :]
        elif url.startswith("http://"):
            url = "ws://" + url[len("http://") :]
        return f"{url}?{query}"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key}"}

    def initial_messages(self) -> list[str]:
        return []

    def text_message(self, text: str) -> str:
        return json.dumps({"type": "Speak", "text": text})

    def finish_message(self) -> str:
        return json.dumps({"type": "Flush"})

    def parse(self, raw: str | bytes) -> NativeProtocolEvent:
        if isinstance(raw, bytes):
            return NativeProtocolEvent(audio=raw, response_started=bool(raw))
        try:
            event = json.loads(raw)
        except json.JSONDecodeError, TypeError, ValueError:
            return NativeProtocolEvent()
        if not isinstance(event, dict):
            return NativeProtocolEvent()
        event_type = event.get("type")
        if event_type == "Error":
            return NativeProtocolEvent(
                error=str(event.get("description") or event.get("code") or event)
            )
        return NativeProtocolEvent(
            audio_done=event_type == "Flushed",
            terminal=event_type == "Flushed",
        )


def _round_ms(value: Optional[float]) -> Optional[float]:
    return round(value, 3) if value is not None else None


def _flatten_exception(exc: BaseException) -> BaseException:
    if not isinstance(exc, BaseExceptionGroup):
        return exc
    leaves: list[BaseException] = []

    def collect(node: BaseException) -> None:
        if isinstance(node, BaseExceptionGroup):
            for child in node.exceptions:
                collect(child)
        else:
            leaves.append(node)

    collect(exc)
    return leaves[0] if leaves else exc


def _map_error(exc: BaseException) -> tuple[int, str]:
    if isinstance(exc, NativeStreamingTTSError):
        return 500, str(exc)
    if isinstance(exc, InvalidStatus):
        body = exc.response.body
        detail = ""
        if isinstance(body, bytes) and body:
            detail = body.decode("utf-8", errors="replace")[:500]
        elif body:
            detail = str(body)[:500]
        message = str(exc) if not detail else f"{exc}: {detail}"
        return exc.response.status_code, message
    if isinstance(exc, TimeoutError):
        return 408, "Streaming TTS request timed out"
    if isinstance(exc, (InvalidHandshake, OSError)):
        return 503, str(exc)
    return 520, str(exc)


class NativeStreamingTTSClient(BaseLLMClient):
    """Shared paced-input/raw-PCM measurement loop for native cloud APIs."""

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
        self._streaming_config = config
        self._protocol = protocol_factory(config, api_key)

    def _connect(self):
        open_timeout = min(self._streaming_config.request_timeout, 30)
        return connect(
            self._protocol.build_ws_url(str(self.api_base)),
            max_size=None,
            compression=None,
            open_timeout=open_timeout,
            additional_headers=self._protocol.headers(),
        )

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
                error_msg="No TEXT channel in request for streaming TTS",
                client_completed_at=time.monotonic(),
            )

        input_text = text_content.input_text
        pacing = self._streaming_config.pacing
        segments = segment_text(input_text, pacing.tokens_per_delta)
        pacer = TextDeltaPacer(pacing, seed=pacing.seed + request.id)
        input_tokens = text_content.target_prompt_tokens or sum(
            segment.n_tokens for segment in segments
        )

        audio_chunks: list[bytes] = []
        audio_chunk_ts: list[list[float]] = []
        text_delta_ts: list[list[float]] = []
        ttfc: Optional[float] = None
        ws_connect_latency: Optional[float] = None
        session_ready_offset: Optional[float] = None
        response_trigger_offset: Optional[float] = None
        response_created_offset: Optional[float] = None
        input_complete_offset: Optional[float] = None
        audio_done_offset: Optional[float] = None
        response_done_offset: Optional[float] = None

        sent_fired = False

        def fire_sent_once() -> None:
            nonlocal sent_fired
            if not sent_fired and on_request_sent is not None:
                on_request_sent()
                sent_fired = True

        t_start = time.monotonic()

        async def send_loop(ws) -> None:
            nonlocal input_complete_offset, response_trigger_offset
            if pacer.initial_delay_s > 0:
                await asyncio.sleep(pacer.initial_delay_s)
            deadline = time.monotonic()
            for segment in segments:
                deadline += pacer.next_gap()
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                offset_ms = (time.monotonic() - t_start) * 1000
                if response_trigger_offset is None:
                    # Native streaming APIs begin accepting synthesis work with
                    # the first real text message. Protocol setup payloads such
                    # as ElevenLabs' single-space initializer are deliberately
                    # excluded from this semantic trigger.
                    response_trigger_offset = offset_ms
                await ws.send(self._protocol.text_message(segment.text))
                text_delta_ts.append([offset_ms, segment.n_chars])
            input_complete_offset = (time.monotonic() - t_start) * 1000
            await ws.send(self._protocol.finish_message())

        async def recv_loop(ws) -> None:
            nonlocal ttfc, session_ready_offset, response_created_offset
            nonlocal audio_done_offset, response_done_offset
            while True:
                raw = await ws.recv()
                wire_offset_ms = (time.monotonic() - t_start) * 1000
                event = self._protocol.parse(raw)
                if event.error:
                    raise NativeStreamingTTSError(event.error)
                if event.ready and session_ready_offset is None:
                    session_ready_offset = wire_offset_ms
                if event.response_started and response_created_offset is None:
                    response_created_offset = wire_offset_ms
                if event.audio:
                    playable_offset_ms = (time.monotonic() - t_start) * 1000
                    audio_chunks.append(event.audio)
                    audio_chunk_ts.append([playable_offset_ms, len(event.audio)])
                    if ttfc is None:
                        ttfc = wire_offset_ms
                    if response_created_offset is None:
                        response_created_offset = wire_offset_ms
                    fire_sent_once()
                if event.audio_done and audio_done_offset is None:
                    audio_done_offset = wire_offset_ms
                if event.terminal:
                    response_done_offset = wire_offset_ms
                    return

        error_code: Optional[int] = None
        error_msg: Optional[str] = None
        try:
            async with asyncio.timeout(self._streaming_config.request_timeout):
                async with self._connect() as ws:
                    ws_connect_latency = (time.monotonic() - t_start) * 1000
                    for message in self._protocol.initial_messages():
                        await ws.send(message)
                    if not self._protocol.has_ready_event:
                        session_ready_offset = (time.monotonic() - t_start) * 1000
                    if on_request_dispatched is not None:
                        on_request_dispatched()
                    async with asyncio.TaskGroup() as task_group:
                        task_group.create_task(send_loop(ws))
                        task_group.create_task(recv_loop(ws))
        except TimeoutError:
            error_code = 408
            error_msg = "Streaming TTS request timed out"
        except Exception as exc:  # noqa: BLE001 - converted to request result.
            error_code, error_msg = _map_error(_flatten_exception(exc))
            logger.warning(
                "%s streaming TTS error: (%s) %s",
                self._protocol.provider,
                error_code,
                error_msg,
            )

        completed_at = time.monotonic()
        total_latency_ms = (completed_at - t_start) * 1000
        success = error_code is None and error_msg is None
        fire_sent_once()

        metrics = {
            "audio_task": AudioTask.TTS,
            AudioMetricKey.PROVIDER.value: self._protocol.provider,
            AudioMetricKey.PROVIDER_MODEL.value: self._streaming_config.model,
            AudioMetricKey.PROVIDER_PROTOCOL.value: self._protocol.protocol_name,
            AudioMetricKey.TTFC.value: round(ttfc or 0.0, 3),
            AudioMetricKey.END_TO_END_LATENCY.value: round(total_latency_ms, 3),
            AudioMetricKey.CHUNK_COUNT.value: len(audio_chunks),
            AudioMetricKey.RAW_PCM.value: True,
            AudioMetricKey.SAMPLE_RATE.value: self._streaming_config.sample_rate,
            AudioMetricKey.INPUT_CHARS.value: len(input_text),
            AudioMetricKey.INPUT_TOKENS.value: input_tokens,
            AudioMetricKey.INPUT_TEXT.value: input_text,
            AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value: text_delta_ts,
            AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value: audio_chunk_ts,
            AudioMetricKey.WS_CONNECT_LATENCY_MS.value: _round_ms(ws_connect_latency),
            AudioMetricKey.SESSION_READY_OFFSET_MS.value: _round_ms(
                session_ready_offset
            ),
            AudioMetricKey.RESPONSE_TRIGGER_OFFSET_MS.value: _round_ms(
                response_trigger_offset
            ),
            AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value: _round_ms(
                input_complete_offset
            ),
            AudioMetricKey.RESPONSE_CREATED_OFFSET_MS.value: _round_ms(
                response_created_offset
            ),
            AudioMetricKey.AUDIO_DONE_OFFSET_MS.value: _round_ms(audio_done_offset),
            AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value: _round_ms(
                response_done_offset
            ),
        }

        channels: dict = {}
        if success or audio_chunks or text_delta_ts:
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=b"".join(audio_chunks),
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


class ElevenLabsStreamingTTSClient(NativeStreamingTTSClient):
    def __init__(self, config: "ElevenLabsStreamingTTSClientConfig", **kwargs) -> None:
        super().__init__(config, ElevenLabsStreamingProtocol)


class DeepgramFluxStreamingTTSClient(NativeStreamingTTSClient):
    def __init__(
        self, config: "DeepgramFluxStreamingTTSClientConfig", **kwargs
    ) -> None:
        super().__init__(config, DeepgramFluxStreamingProtocol)


class DeepgramAuraStreamingTTSClient(NativeStreamingTTSClient):
    def __init__(
        self, config: "DeepgramAuraStreamingTTSClientConfig", **kwargs
    ) -> None:
        super().__init__(config, DeepgramAuraStreamingProtocol)
