import json
from typing import Optional

from vidhi import BasePolyConfig, field, frozen_dataclass

from veeksha.core.audio_contract import DEFAULT_AUDIO_SAMPLE_RATE
from veeksha.core.tokenizer import (
    TokenizerProvider,
    build_hf_tokenizer_handle_from_model,
    build_word_split_tokenizer_provider,
)
from veeksha.logger import init_logger
from veeksha.types import ChannelModality, ClientType

logger = init_logger(__name__)


@frozen_dataclass
class BaseClientConfig(BasePolyConfig):
    """LLM client configuration (OpenAI-compatible API)."""

    api_base: Optional[str] = field(
        None, help="API base URL. Defaults to OPENAI_API_BASE env var."
    )
    api_key: Optional[str] = field(
        None, help="API key. Defaults to OPENAI_API_KEY env var."
    )
    model: str = field(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        help="The model to use for this load test.",
    )
    address_append_value: str = field(
        "chat/completions", help="The address append value for the LLM API."
    )
    request_timeout: int = field(
        300, help="The timeout for each request to the LLM API (in seconds)."
    )
    additional_sampling_params: str = field(
        "{}",
        help="Additional sampling params to send with each request to the LLM API.",
    )

    def __post_init__(self):
        self.additional_sampling_params_dict = {}
        if self.additional_sampling_params:
            self.additional_sampling_params_dict = json.loads(
                self.additional_sampling_params
            )

    def build_tokenizer_provider(self) -> TokenizerProvider:
        """Build a TokenizerProvider for this client config.

        Default implementation uses a HuggingFace tokenizer based on self.model.
        Subclasses can override for non-HF models.
        """
        return TokenizerProvider(
            {ChannelModality.TEXT: build_hf_tokenizer_handle_from_model(self.model)},
            model_name=self.model,
        )


@frozen_dataclass
class OpenAIChatCompletionsClientConfig(BaseClientConfig):
    """OpenAI-compatible Chat Completions client configuration.

    `client.type: openai_chat_completions` uses `/chat/completions` (streaming).
    For per-request routing between chat + completions endpoints (e.g. lm-eval),
    use `client.type: openai_router`.
    """

    max_tokens_param: Optional[str] = field(
        "max_completion_tokens", help="Server parameter name for maximum tokens."
    )
    ignore_eos: bool = field(
        True,
        help="Sets the sampling param ignore_eos for requests to reach the desired max_tokens.",
    )
    min_tokens_param: Optional[str] = field(
        None,
        help="Server parameter name for minimum tokens, usually set if ignore_eos is not available or does no offer enough control over output tokens (see health_check_results.txt). Note: a wrong value might cause requests to fail.",
    )
    use_min_tokens_prompt_fallback: bool = field(
        False,
        help="If True, appends instructions to the prompt to generate at least N tokens (e.g. 'Generate at least 20 tokens'). Useful if the server does not support ignore_eos or min_tokens. Only available on synthetic content generation.",
    )

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.OPENAI_CHAT_COMPLETIONS

    def __post_init__(self):
        super().__post_init__()
        if self.use_min_tokens_prompt_fallback and self.min_tokens_param is None:
            logger.warning(
                "use_min_tokens_prompt_fallback is True but min_tokens_param is None. This will result in no min tokens control."
            )


@frozen_dataclass
class OpenAICompletionsClientConfig(OpenAIChatCompletionsClientConfig):
    """OpenAI Completions client configuration."""

    address_append_value: str = field(
        "completions", help="The address append value for the LLM API."
    )

    max_tokens_param: Optional[str] = field(
        "max_tokens", help="Server parameter name for maximum tokens."
    )

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.OPENAI_COMPLETIONS


@frozen_dataclass
class OpenAIRouterClientConfig(OpenAIChatCompletionsClientConfig):
    """OpenAI-compatible router client configuration.

    This config has the same surface area as `OpenAIChatCompletionsClientConfig`, but the
    corresponding client (`client.type: openai_router`) can route *per request*
    between:
    - `/chat/completions` (streaming)
    - `/completions` (non-stream)

    Routing is controlled by `request.metadata["api_mode"]` (e.g. set by the session generator).

    Note: The two endpoints have different parameter conventions. Use
    `completions_max_tokens_param` to override max tokens for the completions
    endpoint (defaults to "max_tokens"). The chat endpoint uses `max_tokens_param`
    (defaults to "max_completion_tokens").
    """

    completions_max_tokens_param: Optional[str] = field(
        "max_tokens",
        help="Server parameter name for maximum tokens on /completions endpoint. "
        "Defaults to 'max_tokens'. The /chat/completions endpoint uses max_tokens_param instead.",
    )

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.OPENAI_ROUTER


@frozen_dataclass
class TTSClientConfig(BaseClientConfig):
    """Provider-agnostic complete-text HTTP TTS configuration."""

    provider: str = field(
        "",
        help=("HTTP TTS provider. Supported: openai, elevenlabs, deepgram_flux."),
    )
    voice_id: str = field("", help="Voice identifier when required by the provider.")
    sample_rate: int = field(DEFAULT_AUDIO_SAMPLE_RATE, help="Audio sample rate in Hz.")
    chunk_size: int = field(
        1024, help="Read size in bytes for HTTP audio response streaming."
    )
    raw_pcm: bool = field(
        False,
        help="Request raw PCM from providers that support selectable output formats.",
    )
    model: str = field("", help="The TTS model ID.")
    api_key_env: Optional[str] = field(
        None,
        help="Optional provider API-key environment variable override.",
    )
    stability: float = field(0.5, help="ElevenLabs voice stability.")
    similarity_boost: float = field(0.8, help="ElevenLabs similarity boost.")
    speed: float = field(1.0, help="Provider speaking-rate multiplier.")
    apply_text_normalization: str = field(
        "off", help="ElevenLabs text normalization mode: auto | on | off."
    )
    mip_opt_out: bool = field(
        False, help="Opt out of the Deepgram Model Improvement Program."
    )

    _SUPPORTED_PROVIDERS = ("openai", "elevenlabs", "deepgram_flux")
    _VOICE_REQUIRED_PROVIDERS = ("openai", "elevenlabs")

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.TTS

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not self.provider
            and not self.model
            and self.api_base is None
            and not self.voice_id
        ):
            return
        if self.provider not in self._SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported HTTP TTS provider: {self.provider or '<empty>'}. "
                f"Supported: {', '.join(self._SUPPORTED_PROVIDERS)}"
            )
        if not self.model:
            raise ValueError("TTSClientConfig.model is required.")
        if self.api_base is None:
            raise ValueError("TTSClientConfig.api_base is required.")
        if self.provider in self._VOICE_REQUIRED_PROVIDERS and not self.voice_id:
            raise ValueError(
                f"TTSClientConfig.voice_id is required for {self.provider}."
            )
        if self.sample_rate <= 0:
            raise ValueError("TTSClientConfig.sample_rate must be > 0")
        if self.chunk_size <= 0:
            raise ValueError("TTSClientConfig.chunk_size must be > 0")
        if self.provider == "elevenlabs":
            if not 0.7 <= self.speed <= 1.2:
                raise ValueError("ElevenLabs speed must be between 0.7 and 1.2")
            if self.apply_text_normalization not in ("auto", "on", "off"):
                raise ValueError(
                    "apply_text_normalization must be one of ('auto', 'on', 'off')"
                )

    def build_tokenizer_provider(self) -> TokenizerProvider:
        return build_word_split_tokenizer_provider(self.model)


@frozen_dataclass
class TextPacingConfig:
    """LLM decode-rate emulation for paced streaming text input."""

    tokens_per_second: float = field(
        20.0, help="Emulated upstream LLM decode rate (whitespace tokens/sec)."
    )
    tokens_per_delta: int = field(1, help="Whitespace tokens per input append event.")
    gap_distribution: str = field(
        "fixed", help="Inter-delta gap distribution: fixed | poisson."
    )
    initial_delay_s: float = field(
        0.0, help="Delay before the first delta (upstream TTFT emulation)."
    )
    seed: int = field(
        42,
        help="Base seed for per-request gap jitter (request seed = seed + request_id).",
    )

    def __post_init__(self) -> None:
        if self.tokens_per_second <= 0:
            raise ValueError("TextPacingConfig.tokens_per_second must be > 0")
        if self.tokens_per_delta < 1:
            raise ValueError("TextPacingConfig.tokens_per_delta must be >= 1")
        if self.gap_distribution not in ("fixed", "poisson"):
            raise ValueError(
                "TextPacingConfig.gap_distribution must be one of "
                f"('fixed', 'poisson'), got '{self.gap_distribution}'"
            )
        if self.initial_delay_s < 0:
            raise ValueError("TextPacingConfig.initial_delay_s must be >= 0")


@frozen_dataclass
class StreamingTTSClientConfig(BaseClientConfig):
    """Provider-agnostic paced text-in/audio-out WebSocket TTS configuration."""

    provider: str = field(
        "",
        help=(
            "Streaming TTS provider. Supported: openai_realtime, vajra, "
            "elevenlabs, deepgram_flux, deepgram_aura."
        ),
    )
    voice_id: str = field("", help="Optional provider voice identifier.")
    sample_rate: int = field(DEFAULT_AUDIO_SAMPLE_RATE, help="PCM sample rate in Hz.")
    model: str = field("", help="The streaming TTS model ID.")
    api_key_env: Optional[str] = field(
        None,
        help="Optional provider API-key environment variable override.",
    )
    pacing: TextPacingConfig = field(
        default_factory=TextPacingConfig,
        help="Upstream LLM text pacing configuration.",
    )
    input_output_mode: str = field(
        "complete_text",
        help=(
            "Explicit response scheduling: complete_text or duplex. Providers "
            "that synthesize on text receipt use their native trigger behavior."
        ),
    )
    duplex_start_after_tokens: int = field(
        1,
        help="Input tokens before an explicit duplex response trigger.",
    )
    language: Optional[str] = field(
        None, help="Optional language for protocols that support it."
    )
    instructions: Optional[str] = field(
        None, help="Optional synthesis instructions for protocols that support them."
    )
    task_type: Optional[str] = field(None, help="Optional provider task type.")
    chunk_length_schedule: list[int] = field(
        default_factory=lambda: [120, 160, 250, 290],
        help="ElevenLabs character thresholds for audio generation.",
    )
    stability: float = field(0.5, help="ElevenLabs voice stability.")
    similarity_boost: float = field(0.8, help="ElevenLabs similarity boost.")
    speed: float = field(1.0, help="Provider speaking-rate multiplier.")
    auto_mode: bool = field(False, help="Enable ElevenLabs automatic chunk scheduling.")
    apply_text_normalization: str = field(
        "off", help="ElevenLabs text normalization mode: auto | on | off."
    )
    mip_opt_out: bool = field(
        False, help="Opt out of the Deepgram Model Improvement Program."
    )

    _SUPPORTED_PROVIDERS = (
        "openai_realtime",
        "vajra",
        "elevenlabs",
        "deepgram_flux",
        "deepgram_aura",
    )

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.STREAMING_TTS

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not self.provider
            and not self.model
            and self.api_base is None
            and not self.voice_id
        ):
            return
        if self.provider not in self._SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported streaming TTS provider: {self.provider or '<empty>'}. "
                f"Supported: {', '.join(self._SUPPORTED_PROVIDERS)}"
            )
        if not self.model:
            raise ValueError("StreamingTTSClientConfig.model is required.")
        if self.api_base is None:
            raise ValueError("StreamingTTSClientConfig.api_base is required.")
        if self.provider == "elevenlabs" and not self.voice_id:
            raise ValueError(
                "StreamingTTSClientConfig.voice_id is required for elevenlabs."
            )
        if self.sample_rate <= 0:
            raise ValueError("StreamingTTSClientConfig.sample_rate must be > 0")
        if self.input_output_mode not in ("complete_text", "duplex"):
            raise ValueError(
                "StreamingTTSClientConfig.input_output_mode must be one of "
                "('complete_text', 'duplex')"
            )
        if self.duplex_start_after_tokens < 1:
            raise ValueError(
                "StreamingTTSClientConfig.duplex_start_after_tokens must be >= 1"
            )
        if self.provider == "elevenlabs":
            if not self.chunk_length_schedule or any(
                value <= 0 for value in self.chunk_length_schedule
            ):
                raise ValueError("chunk_length_schedule must contain positive values")
            if self.chunk_length_schedule != sorted(self.chunk_length_schedule):
                raise ValueError("chunk_length_schedule must be non-decreasing")
            if not 0.7 <= self.speed <= 1.2:
                raise ValueError("ElevenLabs speed must be between 0.7 and 1.2")
            if self.apply_text_normalization not in ("auto", "on", "off"):
                raise ValueError(
                    "apply_text_normalization must be one of ('auto', 'on', 'off')"
                )
        if self.provider == "deepgram_aura" and not 0.7 <= self.speed <= 1.5:
            raise ValueError("Deepgram Aura speed must be between 0.7 and 1.5")

    def build_tokenizer_provider(self) -> TokenizerProvider:
        return build_word_split_tokenizer_provider(self.model)


@frozen_dataclass
class STTClientConfig(BaseClientConfig):
    """STT client configuration for realtime streaming speech-to-text APIs."""

    provider: str = field(
        "",
        help=(
            "STT provider name. Supported: 'vajra_openai_realtime', " "'vllm_realtime'."
        ),
    )
    sample_rate: int = field(16000, help="Expected audio sample rate in Hz.")
    ws_chunk_size: int = field(
        4096,
        help=(
            "Bytes of raw PCM audio per WebSocket message. Client CPU scales "
            "with concurrency * sample_rate * 2 / ws_chunk_size, so prefer "
            "larger chunks at high concurrency."
        ),
    )
    ws_permessage_deflate: bool = field(
        False,
        help=(
            "Negotiate WebSocket permessage-deflate compression. Disabled by "
            "default because base64 PCM is high entropy and compression adds "
            "substantial client and server CPU."
        ),
    )
    ws_realtime_pacing: bool = field(
        False,
        help=(
            "Sleep between WebSocket audio chunks to simulate realtime input. "
            "Enable for live-audio SLO measurements; disable for engine-bound "
            "throughput measurements."
        ),
    )
    ws_ping_interval_s: Optional[int] = field(
        20, help="WebSocket ping interval in seconds; None disables pings."
    )
    ws_ping_timeout_s: Optional[int] = field(
        None,
        help=(
            "WebSocket ping timeout in seconds. None disables keepalive timeout "
            "while preserving request_timeout."
        ),
    )
    model: str = field("", help="The STT model ID.")

    _SUPPORTED_PROVIDERS = ("vllm_realtime", "vajra_openai_realtime")

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.STT

    def __post_init__(self):
        super().__post_init__()

        # Skip validation for the default polymorphic child instantiated by Vidhi.
        if not self.provider and not self.model and self.api_base is None:
            return
        if not self.provider:
            raise ValueError(
                "STTClientConfig.provider is required. "
                f"Supported: {', '.join(self._SUPPORTED_PROVIDERS)}"
            )
        if self.provider not in self._SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported STT provider: {self.provider}. "
                f"Supported: {', '.join(self._SUPPORTED_PROVIDERS)}"
            )
        if not self.model:
            raise ValueError("STTClientConfig.model is required.")
        if self.api_base is None:
            raise ValueError("STTClientConfig.api_base is required.")
        if self.sample_rate <= 0:
            raise ValueError("STTClientConfig.sample_rate must be > 0")
        if self.ws_chunk_size <= 0:
            raise ValueError("STTClientConfig.ws_chunk_size must be > 0")
        if self.ws_ping_interval_s is not None and self.ws_ping_interval_s <= 0:
            raise ValueError("STTClientConfig.ws_ping_interval_s must be > 0 or None")
        if self.ws_ping_timeout_s is not None and self.ws_ping_timeout_s <= 0:
            raise ValueError("STTClientConfig.ws_ping_timeout_s must be > 0 or None")

    def build_tokenizer_provider(self) -> TokenizerProvider:
        """STT models use a simple word-split tokenizer."""
        return build_word_split_tokenizer_provider(self.model)
