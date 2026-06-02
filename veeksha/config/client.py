import json
from dataclasses import field
from typing import Optional

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.logger import init_logger
from veeksha.types import ClientType

logger = init_logger(__name__)


@frozen_dataclass(allow_from_file=True)
class BaseClientConfig(BasePolyConfig):
    api_base: Optional[str] = field(
        default=None,
        metadata={"help": "API base URL. Defaults to OPENAI_API_BASE env var."},
    )
    api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key. Defaults to OPENAI_API_KEY env var."},
    )
    model: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "The model to use for this load test."},
    )
    address_append_value: str = field(
        default="chat/completions",
        metadata={"help": "The address append value for the LLM API."},
    )
    request_timeout: int = field(
        default=300,
        metadata={"help": "The timeout for each request to the LLM API (in seconds)."},
    )
    additional_sampling_params: str = field(
        default="{}",
        metadata={
            "help": "Additional sampling params to send with each request to the LLM API."
        },
    )

    def __post_init__(self):
        self.additional_sampling_params_dict = {}
        if self.additional_sampling_params:
            self.additional_sampling_params_dict = json.loads(
                self.additional_sampling_params
            )

    def build_tokenizer_provider(self):
        """Build a TokenizerProvider for this client config.

        Default implementation uses a HuggingFace tokenizer based on self.model.
        Subclasses can override for non-HF models.
        """
        from veeksha.core.tokenizer import (
            TokenizerProvider,
            build_hf_tokenizer_handle_from_model,
        )
        from veeksha.types import ChannelModality

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
        default="max_completion_tokens",
        metadata={"help": "Server parameter name for maximum tokens."},
    )
    ignore_eos: bool = field(
        default=True,
        metadata={
            "help": "Sets the sampling param ignore_eos for requests to reach the desired max_tokens."
        },
    )
    min_tokens_param: Optional[str] = field(
        default=None,
        metadata={
            "help": "Server parameter name for minimum tokens, usually set if ignore_eos is not available or does no offer enough control over output tokens (see health_check_results.txt). Note: a wrong value might cause requests to fail."
        },
    )
    use_min_tokens_prompt_fallback: bool = field(
        default=False,
        metadata={
            "help": "If True, appends instructions to the prompt to generate at least N tokens (e.g. 'Generate at least 20 tokens'). Useful if the server does not support ignore_eos or min_tokens. Only available on synthetic content generation."
        },
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
        default="completions",
        metadata={"help": "The address append value for the LLM API."},
    )

    max_tokens_param: Optional[str] = field(
        default="max_tokens",
        metadata={"help": "Server parameter name for maximum tokens."},
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
        default="max_tokens",
        metadata={
            "help": "Server parameter name for maximum tokens on /completions endpoint. "
            "Defaults to 'max_tokens'. The /chat/completions endpoint uses max_tokens_param instead."
        },
    )

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.OPENAI_ROUTER


@frozen_dataclass
class TTSClientConfig(BaseClientConfig):
    """TTS client configuration for Vajra, vLLM Omni and sglang-omni streaming APIs.

    `client.type: tts` sends text to a TTS API and measures audio generation
    metrics from the streamed audio response.
    """

    provider: str = field(
        default="",
        metadata={
            "help": "TTS provider name. Supported: 'vajra', 'vllm_omni', "
            "'sglang_omni'."
        },
    )
    voice_id: str = field(
        default="",
        metadata={
            "help": "Optional voice identifier passed to providers that support it."
        },
    )
    sample_rate: int = field(
        default=24000,
        metadata={"help": "Audio sample rate in Hz."},
    )
    chunk_size: int = field(
        default=1024,
        metadata={"help": "Chunk size in bytes for streaming audio response."},
    )
    raw_pcm: bool = field(
        default=False,
        metadata={
            "help": "Whether the provider returns raw PCM (True) or WAV (False). "
            "vllm_omni always uses raw PCM regardless of this value."
        },
    )
    model: str = field(
        default="",
        metadata={"help": "The TTS model ID."},
    )

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.TTS

    _SUPPORTED_PROVIDERS = ("vajra", "vllm_omni", "sglang_omni")

    def __post_init__(self):
        super().__post_init__()

        # Skip validation when instantiated with defaults by the flat_dataclass
        # framework for non-selected polymorphic children.
        if not self.provider and not self.model:
            return

        if not self.provider:
            raise ValueError(
                "TTSClientConfig.provider is required. "
                f"Supported: {', '.join(self._SUPPORTED_PROVIDERS)}"
            )
        if self.provider not in self._SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported TTS provider: {self.provider}. "
                f"Supported: {', '.join(self._SUPPORTED_PROVIDERS)}"
            )
        if not self.model:
            raise ValueError("TTSClientConfig.model is required.")
        if self.api_base is None:
            raise ValueError("TTSClientConfig.api_base is required.")
        if self.sample_rate <= 0:
            raise ValueError("TTSClientConfig.sample_rate must be > 0")
        if self.chunk_size <= 0:
            raise ValueError("TTSClientConfig.chunk_size must be > 0")
        if self.provider == "vllm_omni" and self.api_key is None:
            object.__setattr__(self, "api_key", "EMPTY")

    def build_tokenizer_provider(self):
        """TTS models use a simple word-split tokenizer."""
        from veeksha.core.tokenizer import TokenizerHandle, TokenizerProvider
        from veeksha.types import ChannelModality

        handle = TokenizerHandle(
            count_tokens=lambda text: len(text.split()),
            decode=lambda ids: " ".join(str(i) for i in ids),
            encode=lambda text: list(range(len(text.split()))),
        )
        return TokenizerProvider(
            {ChannelModality.TEXT: handle},
            model_name=self.model,
        )
