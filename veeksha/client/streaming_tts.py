"""Provider-agnostic WebSocket client for streaming text-to-speech."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote, urlencode, urljoin
from uuid import uuid4

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus

from veeksha.client.base import BaseLLMClient
from veeksha.client.utils import (
    WS_TRANSPORT_ERROR_PRIORITY,
    TextDeltaPacer,
    flatten_ws_exception,
    map_ws_transport_error,
    resolve_provider_api_key,
    segment_text,
    to_websocket_url,
)
from veeksha.core.audio_contract import AudioMetricKey, pcm_bytes_to_duration_ms
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import StreamingTTSClientConfig

logger = init_logger(__name__)
__all__ = ["StreamingTTSClient"]


@dataclass(frozen=True)
class StreamingProtocolEvent:
    """Transport-independent event produced by a provider strategy."""

    audio: bytes = b""
    ready: bool = False
    response_started: bool = False
    audio_done: bool = False
    terminal: bool = False
    sample_rate: int | None = None
    error: str | None = None


class StreamingTTSError(Exception):
    """Fatal provider event received after a WebSocket handshake."""


class _ClientAbort(Exception):
    """Signal a deliberate mid-stream client hang-up.

    This is an expected adversarial outcome, not a server or transport error.
    """


class StreamingProviderProtocol(Protocol):
    """Provider-specific wire behavior behind the shared streaming lifecycle."""

    provider: str
    protocol_name: str
    default_api_key_env: str
    requires_api_key: bool
    has_ready_event: bool
    explicit_response_trigger: bool
    raw_pcm: bool

    def __init__(
        self, config: StreamingTTSClientConfig, api_key: str | None
    ) -> None: ...

    def build_ws_url(self, api_base: str) -> str: ...

    def headers(self) -> dict[str, str]: ...

    def initial_messages(self) -> list[str]: ...

    def text_message(self, text: str) -> str: ...

    def response_trigger_message(self) -> str | None: ...

    def finish_messages(self) -> list[str]: ...

    def parse(self, raw: str | bytes) -> StreamingProtocolEvent: ...


class _ImplicitResponseProtocol:
    """Common scheduling behavior for providers that synthesize on text receipt."""

    explicit_response_trigger = False
    raw_pcm = True

    def response_trigger_message(self) -> None:
        return None


def _decode_json_object(raw: str | bytes) -> dict[str, Any] | None:
    if isinstance(raw, bytes):
        return None
    try:
        event = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return event if isinstance(event, dict) else None


def _extract_realtime_sample_rate(session: dict[str, Any]) -> int | None:
    audio = session.get("audio")
    if not isinstance(audio, dict):
        return None
    output = audio.get("output")
    if not isinstance(output, dict):
        return None
    audio_format = output.get("format")
    if not isinstance(audio_format, dict):
        return None
    sample_rate = audio_format.get("rate")
    return sample_rate if isinstance(sample_rate, int) else None


def _realtime_error_message(event: dict[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message

    response = event.get("response")
    if isinstance(response, dict):
        details = response.get("status_details")
        if isinstance(details, dict):
            detail_error = details.get("error")
            if isinstance(detail_error, dict):
                message = detail_error.get("message")
                if isinstance(message, str) and message:
                    return message
            reason = details.get("reason")
            if isinstance(reason, str) and reason:
                return reason
        status = response.get("status")
        if isinstance(status, str) and status:
            return f"Realtime response ended with status {status}"

    return json.dumps(event)


class OpenAIRealtimeProtocol:
    provider = "openai"
    protocol_name = "realtime"
    default_api_key_env = "OPENAI_API_KEY"
    requires_api_key = False
    has_ready_event = True
    explicit_response_trigger = True
    raw_pcm = True

    def __init__(self, config: StreamingTTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key

    def build_ws_url(self, api_base: str) -> str:
        normalized = to_websocket_url(api_base).rstrip("/")
        path = "realtime" if normalized.endswith("/v1") else "v1/realtime"
        url = urljoin(f"{normalized}/", path)
        return f"{url}?model={quote(self.config.model, safe='')}"

    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def initial_messages(self) -> list[str]:
        output: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": self.config.sample_rate}
        }
        if self.config.voice_id:
            output["voice"] = self.config.voice_id
        session = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {"output": output},
        }
        return [json.dumps({"type": "session.update", "session": session})]

    def text_message(self, text: str) -> str:
        return json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    def response_trigger_message(self) -> str:
        return json.dumps(
            {
                "type": "response.create",
                "response": {"output_modalities": ["audio"]},
            }
        )

    def finish_messages(self) -> list[str]:
        return []

    def parse(self, raw: str | bytes) -> StreamingProtocolEvent:
        event = _decode_json_object(raw)
        if event is None:
            return StreamingProtocolEvent()
        event_type = event.get("type")
        if event_type == "session.updated":
            session = event.get("session")
            sample_rate = (
                _extract_realtime_sample_rate(session)
                if isinstance(session, dict)
                else None
            )
            return StreamingProtocolEvent(ready=True, sample_rate=sample_rate)
        if event_type == "response.created":
            return StreamingProtocolEvent(response_started=True)
        if event_type == "response.output_audio.delta":
            encoded_audio = event.get("delta")
            if not isinstance(encoded_audio, str) or not encoded_audio:
                return StreamingProtocolEvent()
            try:
                audio = base64.b64decode(encoded_audio, validate=True)
            except (ValueError, TypeError) as exc:
                return StreamingProtocolEvent(
                    error=f"Invalid OpenAI Realtime audio: {exc}"
                )
            return StreamingProtocolEvent(audio=audio)
        if event_type == "response.output_audio.done":
            return StreamingProtocolEvent(audio_done=True)
        if event_type == "response.done":
            response = event.get("response")
            if not isinstance(response, dict) or response.get("status") != "completed":
                return StreamingProtocolEvent(error=_realtime_error_message(event))
            return StreamingProtocolEvent(terminal=True)
        if event_type == "error":
            return StreamingProtocolEvent(error=_realtime_error_message(event))
        return StreamingProtocolEvent()


class VajraStreamingProtocol(_ImplicitResponseProtocol):
    provider = "vajra"
    protocol_name = "native_streaming_text"
    default_api_key_env = "OPENAI_API_KEY"
    requires_api_key = False
    # The native protocol has no pre-synthesis session-ready acknowledgement;
    # audio.start is a response event emitted only after input.text.
    has_ready_event = False

    def __init__(self, config: StreamingTTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key

    def build_ws_url(self, api_base: str) -> str:
        normalized = to_websocket_url(api_base).rstrip("/")
        path = (
            "audio/speech/stream"
            if normalized.endswith("/v1")
            else "v1/audio/speech/stream"
        )
        return urljoin(f"{normalized}/", path)

    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def initial_messages(self) -> list[str]:
        message: dict[str, Any] = {
            "type": "session.config",
            "response_format": "pcm",
            "stream_audio": True,
        }
        if self.config.voice_id:
            message["voice"] = self.config.voice_id
        if self.config.language is not None:
            message["language"] = self.config.language
        if self.config.instructions is not None:
            message["instructions"] = self.config.instructions
        if self.config.task_type is not None:
            message["task_type"] = self.config.task_type
        return [json.dumps(message)]

    def text_message(self, text: str) -> str:
        return json.dumps({"type": "input.text", "text": text})

    def finish_messages(self) -> list[str]:
        return [json.dumps({"type": "input.done"})]

    def parse(self, raw: str | bytes) -> StreamingProtocolEvent:
        if isinstance(raw, bytes):
            return StreamingProtocolEvent(audio=raw)
        event = _decode_json_object(raw)
        if event is None:
            return StreamingProtocolEvent()
        event_type = event.get("type")
        if event_type == "audio.start":
            sample_rate = event.get("sample_rate")
            return StreamingProtocolEvent(
                response_started=True,
                sample_rate=sample_rate if isinstance(sample_rate, int) else None,
            )
        if event_type == "audio.done":
            if event.get("error"):
                return StreamingProtocolEvent(
                    error="Vajra TTS stream reported audio.done with error=true"
                )
            return StreamingProtocolEvent(audio_done=True)
        if event_type == "session.done":
            return StreamingProtocolEvent(terminal=True)
        if event_type == "error":
            message = event.get("message")
            return StreamingProtocolEvent(
                error=message if isinstance(message, str) else json.dumps(event)
            )
        return StreamingProtocolEvent()


class ElevenLabsStreamingProtocol(_ImplicitResponseProtocol):
    provider = "elevenlabs"
    protocol_name = "v1_stream_input"
    default_api_key_env = "ELEVENLABS_API_KEY"
    requires_api_key = True
    has_ready_event = False

    def __init__(self, config: StreamingTTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def build_ws_url(self, api_base: str) -> str:
        normalized = api_base.rstrip("/") + "/"
        path = f"v1/text-to-speech/{quote(self.config.voice_id, safe='')}/stream-input"
        parameters = {
            "model_id": self.config.model,
            "output_format": f"pcm_{self.config.sample_rate}",
            "auto_mode": str(self.config.auto_mode).lower(),
            "apply_text_normalization": self.config.apply_text_normalization,
        }
        if self.config.language is not None:
            parameters["language_code"] = self.config.language
        query = urlencode(parameters)
        return f"{to_websocket_url(urljoin(normalized, path))}?{query}"

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

    def finish_messages(self) -> list[str]:
        return [json.dumps({"text": ""})]

    def parse(self, raw: str | bytes) -> StreamingProtocolEvent:
        if isinstance(raw, bytes):
            return StreamingProtocolEvent(error="Unexpected binary ElevenLabs frame")
        event = _decode_json_object(raw)
        if event is None:
            return StreamingProtocolEvent()
        error = event.get("error")
        if error:
            if isinstance(error, dict):
                message = error.get("message") or json.dumps(error)
            else:
                message = str(error)
            return StreamingProtocolEvent(error=message)
        audio = b""
        encoded_audio = event.get("audio")
        if isinstance(encoded_audio, str) and encoded_audio:
            try:
                audio = base64.b64decode(encoded_audio, validate=True)
            except (ValueError, TypeError) as exc:
                return StreamingProtocolEvent(error=f"Invalid ElevenLabs audio: {exc}")
        terminal = bool(event.get("isFinal"))
        return StreamingProtocolEvent(
            audio=audio,
            response_started=bool(audio),
            audio_done=terminal,
            terminal=terminal,
        )


class CartesiaStreamingProtocol(_ImplicitResponseProtocol):
    """Cartesia context protocol with incremental transcript appends."""

    provider = "cartesia"
    protocol_name = "tts_websocket_context"
    default_api_key_env = "CARTESIA_API_KEY"
    requires_api_key = True
    has_ready_event = False

    def __init__(self, config: StreamingTTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""
        self.context_id = str(uuid4())

    def build_ws_url(self, api_base: str) -> str:
        normalized = api_base.rstrip("/") + "/"
        return to_websocket_url(urljoin(normalized, "tts/websocket"))

    def headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "Cartesia-Version": self.config.cartesia_version,
        }

    def initial_messages(self) -> list[str]:
        return []

    def _generation_message(self, transcript: str, *, continuing: bool) -> str:
        payload: dict[str, Any] = {
            "model_id": self.config.model,
            "transcript": transcript,
            "voice": {"mode": "id", "id": self.config.voice_id},
            "context_id": self.context_id,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.config.sample_rate,
            },
            "continue": continuing,
        }
        if self.config.language is not None:
            payload["language"] = self.config.language
        if self.config.max_buffer_delay_ms is not None:
            payload["max_buffer_delay_ms"] = self.config.max_buffer_delay_ms
        return json.dumps(payload)

    def text_message(self, text: str) -> str:
        return self._generation_message(text, continuing=True)

    def finish_messages(self) -> list[str]:
        return [self._generation_message("", continuing=False)]

    def parse(self, raw: str | bytes) -> StreamingProtocolEvent:
        if isinstance(raw, bytes):
            return StreamingProtocolEvent(error="Unexpected binary Cartesia frame")
        event = _decode_json_object(raw)
        if event is None:
            return StreamingProtocolEvent()
        event_type = event.get("type")
        if event_type == "error":
            message = (
                event.get("message")
                or event.get("title")
                or event.get("error_code")
                or json.dumps(event)
            )
            return StreamingProtocolEvent(error=str(message))
        if event_type == "chunk":
            encoded_audio = event.get("data")
            if not isinstance(encoded_audio, str) or not encoded_audio:
                return StreamingProtocolEvent(error="Cartesia chunk omitted audio data")
            try:
                audio = base64.b64decode(encoded_audio, validate=True)
            except (ValueError, TypeError) as exc:
                return StreamingProtocolEvent(error=f"Invalid Cartesia audio: {exc}")
            return StreamingProtocolEvent(audio=audio, response_started=True)
        if event_type == "done" or event.get("done") is True:
            return StreamingProtocolEvent(audio_done=True, terminal=True)
        return StreamingProtocolEvent()


class _DeepgramStreamingProtocol(_ImplicitResponseProtocol):
    provider = "deepgram"
    default_api_key_env = "DEEPGRAM_API_KEY"
    requires_api_key = True
    endpoint = ""
    ready_event: str | None = None
    response_started_event: str | None = None
    terminal_event = ""

    def __init__(self, config: StreamingTTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def _query_parameters(self) -> dict[str, str | int | float]:
        return {
            "model": self.config.model,
            "encoding": "linear16",
            "sample_rate": self.config.sample_rate,
            "mip_opt_out": str(self.config.mip_opt_out).lower(),
        }

    def build_ws_url(self, api_base: str) -> str:
        normalized = api_base.rstrip("/") + "/"
        query = urlencode(self._query_parameters())
        return f"{to_websocket_url(urljoin(normalized, self.endpoint))}?{query}"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key}"}

    def initial_messages(self) -> list[str]:
        return []

    def text_message(self, text: str) -> str:
        return json.dumps({"type": "Speak", "text": text})

    def finish_messages(self) -> list[str]:
        return [json.dumps({"type": "Flush"})]

    def parse(self, raw: str | bytes) -> StreamingProtocolEvent:
        if isinstance(raw, bytes):
            return StreamingProtocolEvent(audio=raw)
        event = _decode_json_object(raw)
        if event is None:
            return StreamingProtocolEvent()
        event_type = event.get("type")
        if event_type == "Error":
            return StreamingProtocolEvent(
                error=str(event.get("description") or event.get("code") or event)
            )
        terminal = event_type == self.terminal_event
        return StreamingProtocolEvent(
            ready=event_type == self.ready_event,
            response_started=event_type == self.response_started_event,
            audio_done=terminal,
            terminal=terminal,
        )


class DeepgramFluxStreamingProtocol(_DeepgramStreamingProtocol):
    protocol_name = "v2_flux_speak"
    endpoint = "v2/speak"
    has_ready_event = True
    ready_event = "Connected"
    response_started_event = "SpeechStarted"
    terminal_event = "SpeechMetadata"


class DeepgramAuraStreamingProtocol(_DeepgramStreamingProtocol):
    protocol_name = "v1_aura_speak"
    endpoint = "v1/speak"
    has_ready_event = False
    terminal_event = "Flushed"

    def _query_parameters(self) -> dict[str, str | int | float]:
        parameters = super()._query_parameters()
        parameters["speed"] = self.config.speed
        return parameters


_STREAMING_PROTOCOLS: dict[str, type[StreamingProviderProtocol]] = {
    "openai_realtime": OpenAIRealtimeProtocol,
    "vajra": VajraStreamingProtocol,
    "elevenlabs": ElevenLabsStreamingProtocol,
    "cartesia": CartesiaStreamingProtocol,
    "deepgram_flux": DeepgramFluxStreamingProtocol,
    "deepgram_aura": DeepgramAuraStreamingProtocol,
}


_ERROR_PRIORITY: tuple[type[BaseException], ...] = (
    StreamingTTSError,
    *WS_TRANSPORT_ERROR_PRIORITY,
)


_ABORT_ERROR_PRIORITY: tuple[type[BaseException], ...] = (
    _ClientAbort,
    *_ERROR_PRIORITY,
)


def _map_error(exc: BaseException) -> tuple[int, str]:
    if isinstance(exc, StreamingTTSError):
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
    return map_ws_transport_error(exc, "Streaming TTS request timed out")


def _round_ms(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


class StreamingTTSClient(BaseLLMClient):
    """Paced text-in/audio-out WebSocket TTS client for every provider."""

    def __init__(self, config: StreamingTTSClientConfig, **kwargs: Any) -> None:
        protocol_class = _STREAMING_PROTOCOLS.get(config.provider)
        if protocol_class is None:
            raise ValueError(f"Unsupported streaming TTS provider: {config.provider}")
        super().__init__(config)
        self.api_key = resolve_provider_api_key(
            config.api_key,
            config.api_key_env,
            protocol_class.default_api_key_env,
            required=protocol_class.requires_api_key,
        )
        self._streaming_config = config
        self._protocol = protocol_class(config, self.api_key)

    def _connect(self, protocol: StreamingProviderProtocol | None = None) -> Any:
        protocol = protocol or self._protocol
        open_timeout = min(self._streaming_config.request_timeout, 30)
        return connect(
            protocol.build_ws_url(str(self.api_base)),
            max_size=None,
            compression=None,
            open_timeout=open_timeout,
            additional_headers=protocol.headers(),
        )

    def _request_language(self, metadata: dict[str, Any]) -> str | None:
        """Resolve one request's language without mutating shared client state."""

        mode = self._streaming_config.language_mode
        if mode == "auto":
            return None
        if mode == "fixed":
            return self._streaming_config.language

        key = self._streaming_config.language_metadata_key
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Streaming TTS request metadata must contain a non-empty {key!r} "
                "string when language_mode=request_metadata"
            )
        language = value.strip()
        supported = {
            item.strip().casefold()
            for item in self._streaming_config.supported_languages
        }
        if supported and language.casefold() not in supported:
            raise ValueError(
                f"Streaming TTS target {self._streaming_config.model!r} does not "
                f"declare support for request language {language!r}"
            )
        return language

    async def measure_websocket_rtt_ms(self, samples: int = 5) -> list[float]:
        """Measure ping/pong RTT on independent provider WebSocket connections.

        The probe authenticates against the configured streaming-TTS endpoint
        but doesn't send provider application messages or synthesis text.
        Independent connections match the benchmark's one-connection-per-request
        lifecycle while excluding DNS, TCP, TLS, and WebSocket handshake time
        from the returned measurements.
        """
        if samples < 1:
            raise ValueError("samples must be >= 1")

        pong_timeout_s = min(float(self._streaming_config.request_timeout), 10.0)
        rtt_samples_ms: list[float] = []
        for _ in range(samples):
            async with self._connect() as websocket:
                pong_received = await websocket.ping()
                rtt_s = await asyncio.wait_for(
                    pong_received,
                    timeout=pong_timeout_s,
                )
                rtt_samples_ms.append(rtt_s * 1000)
        return rtt_samples_ms

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
                error_msg="No TEXT channel in request for streaming TTS",
                client_completed_at=time.monotonic(),
            )

        input_text = text_content.input_text
        try:
            request_language = self._request_language(request.metadata)
        except ValueError as error:
            if on_request_dispatched is not None:
                on_request_dispatched()
            if on_request_sent is not None:
                on_request_sent()
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=400,
                error_msg=str(error),
                client_completed_at=time.monotonic(),
            )
        # Protocol instances are request-scoped so providers such as Cartesia can
        # safely retain per-utterance continuation state under concurrency.
        request_config = replace(self._streaming_config, language=request_language)
        protocol = type(self._protocol)(request_config, self.api_key)
        if self._streaming_config.strict_audio_contract and not protocol.raw_pcm:
            raise ValueError(
                f"{protocol.provider} does not provide raw PCM required by the "
                "strict streaming TTS audio contract"
            )
        pacing = self._streaming_config.pacing
        if self._streaming_config.input_output_mode == "complete_text":
            # ``streaming_tts`` describes the WebSocket transport, not the
            # input interaction.  A complete-text benchmark must make the
            # entire prompt eligible for synthesis in one append and must not
            # accidentally emulate an upstream LLM token cadence.
            segments = segment_text(input_text, max(1, len(input_text.split())))
        else:
            segments = segment_text(input_text, pacing.tokens_per_delta)
        pacer = TextDeltaPacer(pacing, seed=pacing.seed + request.id)
        input_tokens = text_content.target_prompt_tokens or sum(
            segment.n_tokens for segment in segments
        )

        abort_config = self._streaming_config.abort
        abort_input_after: int | None = None
        abort_audio_ms: float | None = None
        abort_wall_s: float | None = None
        if abort_config.selects(session_id):
            if abort_config.trigger == "input_fraction":
                abort_input_after = max(
                    1, math.ceil(abort_config.value * len(segments))
                )
            elif abort_config.trigger == "audio_ms":
                abort_audio_ms = abort_config.value
            elif abort_config.trigger == "wall_clock_s":
                abort_wall_s = abort_config.value

        audio_chunks: list[bytes] = []
        audio_chunk_timestamps: list[list[float | int]] = []
        text_delta_timestamps: list[list[float | int]] = []
        ttfc: float | None = None
        ws_connect_latency: float | None = None
        session_ready_offset: float | None = None
        response_trigger_offset: float | None = None
        response_created_offset: float | None = None
        input_complete_offset: float | None = None
        audio_done_offset: float | None = None
        response_done_offset: float | None = None
        sample_rate = self._streaming_config.sample_rate
        sent_fired = False
        aborted = False
        stream_completed = asyncio.Event()
        session_ready = asyncio.Event()
        start = time.monotonic()

        def fire_sent_once() -> None:
            nonlocal sent_fired
            if not sent_fired and on_request_sent is not None:
                on_request_sent()
                sent_fired = True

        async def send_loop(websocket: ClientConnection) -> None:
            nonlocal input_complete_offset, response_trigger_offset
            # Application-level session readiness is separate from the WS
            # handshake. Do not let text or the synthesis trigger race provider
            # configuration, and keep trigger-to-audio latency independent of
            # connection/session setup.
            await session_ready.wait()
            paced_input = self._streaming_config.input_output_mode == "duplex"
            if paced_input and pacer.initial_delay_s > 0:
                await asyncio.sleep(pacer.initial_delay_s)
            deadline = time.monotonic()
            response_triggered = False
            sent_words = 0

            for segment_idx, segment in enumerate(segments):
                if paced_input:
                    deadline += pacer.next_gap()
                    sleep_s = deadline - time.monotonic()
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)
                offset_ms = (time.monotonic() - start) * 1000
                if not protocol.explicit_response_trigger and not response_triggered:
                    # Native streaming providers treat the first real text
                    # append as the synthesis trigger. Record it before the
                    # await so a very fast response cannot race the timestamp.
                    response_trigger_offset = offset_ms
                    response_triggered = True

                await websocket.send(protocol.text_message(segment.text))
                text_delta_timestamps.append([offset_ms, segment.n_chars])
                sent_words += segment.n_tokens

                if (
                    protocol.explicit_response_trigger
                    and not response_triggered
                    and self._streaming_config.input_output_mode == "duplex"
                    and sent_words >= self._streaming_config.duplex_start_after_tokens
                ):
                    response_trigger_offset = (time.monotonic() - start) * 1000
                    trigger_message = protocol.response_trigger_message()
                    if trigger_message is not None:
                        await websocket.send(trigger_message)
                    response_triggered = True

                if (
                    abort_input_after is not None
                    and segment_idx + 1 >= abort_input_after
                ):
                    raise _ClientAbort()

            input_complete_offset = (time.monotonic() - start) * 1000
            if protocol.explicit_response_trigger and not response_triggered:
                response_trigger_offset = (time.monotonic() - start) * 1000
                trigger_message = protocol.response_trigger_message()
                if trigger_message is not None:
                    await websocket.send(trigger_message)
                response_triggered = True

            for message in protocol.finish_messages():
                await websocket.send(message)

        async def recv_loop(websocket: ClientConnection) -> None:
            nonlocal ttfc, session_ready_offset, response_created_offset
            nonlocal audio_done_offset, response_done_offset, sample_rate
            received_audio_ms = 0.0
            while True:
                raw = await websocket.recv()
                wire_offset_ms = (time.monotonic() - start) * 1000
                event = protocol.parse(raw)
                if event.error:
                    raise StreamingTTSError(event.error)
                if event.ready and session_ready_offset is None:
                    session_ready_offset = wire_offset_ms
                    session_ready.set()
                if event.response_started and response_created_offset is None:
                    response_created_offset = wire_offset_ms
                if event.sample_rate is not None and event.sample_rate > 0:
                    if event.sample_rate != sample_rate:
                        if self._streaming_config.strict_audio_contract:
                            raise StreamingTTSError(
                                f"{protocol.provider} sent sample_rate="
                                f"{event.sample_rate}; benchmark requires "
                                f"{sample_rate}"
                            )
                        logger.warning(
                            "%s sent sample_rate=%d (configured %d); "
                            "using the provider value",
                            protocol.provider,
                            event.sample_rate,
                            sample_rate,
                        )
                    sample_rate = event.sample_rate
                if event.audio:
                    if (
                        self._streaming_config.strict_audio_contract
                        and len(event.audio) % 2
                    ):
                        raise StreamingTTSError(
                            f"{protocol.provider} sent an odd-length PCM16 payload "
                            f"({len(event.audio)} bytes)"
                        )
                    playable_offset_ms = (time.monotonic() - start) * 1000
                    audio_chunks.append(event.audio)
                    audio_chunk_timestamps.append(
                        [playable_offset_ms, len(event.audio)]
                    )
                    if ttfc is None:
                        if response_trigger_offset is None:
                            raise StreamingTTSError(
                                "received audio before the synthesis trigger"
                            )
                        ttfc = wire_offset_ms - response_trigger_offset
                    if response_created_offset is None:
                        response_created_offset = wire_offset_ms
                    fire_sent_once()
                    if abort_audio_ms is not None:
                        received_audio_ms += pcm_bytes_to_duration_ms(
                            len(event.audio), sample_rate
                        )
                        if received_audio_ms >= abort_audio_ms:
                            raise _ClientAbort()
                if event.audio_done and audio_done_offset is None:
                    audio_done_offset = wire_offset_ms
                if event.terminal:
                    response_done_offset = wire_offset_ms
                    stream_completed.set()
                    return

        async def abort_watchdog(delay_s: float) -> None:
            try:
                await asyncio.wait_for(stream_completed.wait(), timeout=delay_s)
            except TimeoutError as error:
                raise _ClientAbort() from error

        error_code: int | None = None
        error_msg: str | None = None
        try:
            async with asyncio.timeout(self._streaming_config.request_timeout):
                async with self._connect(protocol) as websocket:
                    ws_connect_latency = (time.monotonic() - start) * 1000
                    for message in protocol.initial_messages():
                        await websocket.send(message)
                    if not protocol.has_ready_event:
                        session_ready_offset = (time.monotonic() - start) * 1000
                        session_ready.set()
                    if on_request_dispatched is not None:
                        on_request_dispatched()
                    async with asyncio.TaskGroup() as task_group:
                        task_group.create_task(send_loop(websocket))
                        task_group.create_task(recv_loop(websocket))
                        if abort_wall_s is not None:
                            task_group.create_task(abort_watchdog(abort_wall_s))
        except Exception as exc:
            flattened = flatten_ws_exception(exc, _ABORT_ERROR_PRIORITY)
            if isinstance(flattened, _ClientAbort):
                aborted = True
                logger.debug(
                    "%s streaming TTS request_id=%d session_id=%d aborted "
                    "mid-stream (trigger=%s value=%s)",
                    protocol.provider,
                    request.id,
                    session_id,
                    abort_config.trigger,
                    abort_config.value,
                )
            else:
                error_code, error_msg = _map_error(flattened)
                logger.warning(
                    "%s streaming TTS error: (%s) %s",
                    protocol.provider,
                    error_code,
                    error_msg,
                )

        completed_at = time.monotonic()
        latency_ms = (completed_at - start) * 1000
        if error_code is None and not aborted and not audio_chunks:
            error_code = 502
            error_msg = f"{protocol.provider} completed the TTS stream without audio"
        success = error_code is None and error_msg is None
        fire_sent_once()

        metrics = {
            "audio_task": AudioTask.TTS,
            AudioMetricKey.PROVIDER.value: protocol.provider,
            AudioMetricKey.PROVIDER_MODEL.value: self._streaming_config.model,
            AudioMetricKey.PROVIDER_PROTOCOL.value: protocol.protocol_name,
            AudioMetricKey.TTFC.value: (round(ttfc, 3) if ttfc is not None else None),
            AudioMetricKey.END_TO_END_LATENCY.value: round(latency_ms, 3),
            AudioMetricKey.CHUNK_COUNT.value: len(audio_chunks),
            AudioMetricKey.RAW_PCM.value: protocol.raw_pcm,
            AudioMetricKey.SAMPLE_RATE.value: sample_rate,
            AudioMetricKey.AUDIO_ENCODING.value: "pcm_s16le",
            AudioMetricKey.AUDIO_CHANNELS.value: 1,
            AudioMetricKey.INPUT_CHARS.value: len(input_text),
            AudioMetricKey.INPUT_TOKENS.value: input_tokens,
            AudioMetricKey.INPUT_TEXT.value: input_text,
            AudioMetricKey.TEXT_PACING_UNIT.value: (
                "whitespace_word"
                if self._streaming_config.input_output_mode == "duplex"
                else "complete_text"
            ),
            AudioMetricKey.TEXT_PACING_RATE.value: (
                pacing.tokens_per_second
                if self._streaming_config.input_output_mode == "duplex"
                else None
            ),
            "language_routing_mode": self._streaming_config.language_mode,
            "provider_language_value": request_language,
            AudioMetricKey.ABORTED.value: aborted,
            AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value: text_delta_timestamps,
            AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value: audio_chunk_timestamps,
            AudioMetricKey.WS_CONNECT_LATENCY_MS.value: _round_ms(ws_connect_latency),
            AudioMetricKey.SESSION_READY_OFFSET_MS.value: _round_ms(
                session_ready_offset
            ),
            AudioMetricKey.RESPONSE_TRIGGER_OFFSET_MS.value: _round_ms(
                response_trigger_offset
            ),
            AudioMetricKey.RESPONSE_CREATED_OFFSET_MS.value: _round_ms(
                response_created_offset
            ),
            AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value: _round_ms(
                input_complete_offset
            ),
            AudioMetricKey.AUDIO_DONE_OFFSET_MS.value: _round_ms(audio_done_offset),
            AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value: _round_ms(
                response_done_offset
            ),
        }
        for key, value in request.metadata.items():
            metrics.setdefault(key, value)

        channels: dict[ChannelModality, ChannelResponse] = {}
        if success or audio_chunks or text_delta_timestamps or aborted:
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
