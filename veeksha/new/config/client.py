import json
from dataclasses import field
from typing import Optional

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.logger import init_logger
from veeksha.new.types import ClientType

logger = init_logger(__name__)


@frozen_dataclass
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
class OpenAIChatClientConfig(BaseClientConfig):
    """OpenAI Chat Completions client configuration."""

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
            "help": "If True, appends instructions to the prompt to generate at least N tokens (e.g. 'Generate at least 20 tokens'). Useful if the server does not support a min tokens parameter."
        },
    )

    @classmethod
    def get_type(cls) -> ClientType:
        return ClientType.OPENAI_CHAT

    def __post_init__(self):
        if self.use_min_tokens_prompt_fallback and self.min_tokens_param is None:
            logger.warning(
                "use_min_tokens_prompt_fallback is True but min_tokens_param is None. This will result in no min tokens control."
            )
