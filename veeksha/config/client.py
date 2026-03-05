import json
import os
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
    """TTS client configuration for Deepgram/ElevenLabs streaming APIs.

    `client.type: tts` sends text to a TTS API and measures audio generation metrics.
    """

    provider: str = field(
        default="",
        metadata={
            "help": "TTS provider name. "
            "Supported: 'deepgram', 'elevenlabs', 'vajra', 'voxserve', 'vllm_omni'."
        },
    )
    voice_id: str = field(
        default="",
        metadata={
            "help": "Voice identifier passed to the TTS provider. "
            "Required for ElevenLabs (e.g. 'JBFqnCBsd6RMkjVDRZzb')."
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
            "help": "Whether the provider returns raw PCM (True) or WAV (False)."
        },
    )
    model: str = field(
        default="",
        metadata={
            "help": "The TTS model ID (e.g. 'aura-2-thalia-en', 'eleven_turbo_v2_5')."
        },
    )

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.TTS

    _SUPPORTED_PROVIDERS = ("deepgram", "elevenlabs", "vajra", "voxserve", "vllm_omni")

    def __post_init__(self):
        super().__post_init__()

        # --- Required field checks ---
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
            raise ValueError(
                "TTSClientConfig.model is required "
                "(e.g. 'aura-2-thalia-en', 'eleven_turbo_v2_5')."
            )
        if self.provider == "elevenlabs" and not self.voice_id:
            raise ValueError(
                "TTSClientConfig.voice_id is required for the ElevenLabs provider."
            )
        if self.api_base is None:
            raise ValueError("TTSClientConfig.api_base is required.")
        if self.api_key is None:
            env_map = {
                "deepgram": "DEEPGRAM_API_KEY",
                "elevenlabs": "ELEVENLABS_API_KEY",
            }
            env_var = env_map.get(self.provider)
            if env_var:
                key = os.environ.get(env_var)
                if key:
                    object.__setattr__(self, "api_key", key)

        # Auto-set raw_pcm for ElevenLabs (returns raw PCM, not WAV)
        if self.provider == "elevenlabs" and not self.raw_pcm:
            object.__setattr__(self, "raw_pcm", True)

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
