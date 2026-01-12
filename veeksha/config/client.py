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
    min_tokens_param: Optional[str] = field(
        default="min_tokens",
        metadata={
            "help": "Server parameter name for minimum tokens. If your server supports min tokens control via a parameter, specify its name here."
        },
    )
    use_min_tokens_prompt_fallback: bool = field(
        default=False,
        metadata={
            "help": "If True, appends instructions to the prompt to generate at least N tokens (e.g. 'Generate at least 20 tokens'). Useful if the server does not support a min tokens parameter. Only available on synthetic content generation."
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
