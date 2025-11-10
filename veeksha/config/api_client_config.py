import json
from dataclasses import field
from typing import Any, Dict, Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


@frozen_dataclass
class BaseApiClientConfig(BasePolyConfig):
    model: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "The model to use for this load test."},
    )
    override_tokenizer: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for this load test. By default, the tokenizer is inferred from the model."
        },
    )
    num_concurrent_requests: int = field(
        default=5,
        metadata={"help": "The number of concurrent requests to send per client."},
    )
    request_timeout: int = field(
        default=60,
        metadata={"help": "The timeout for each request to the LLM API (in seconds)."},
    )
    api_url: Optional[str] = field(
        default="http://localhost:8000/v1",
        metadata={"help": "The API endpoint URL."},
    )
    api_key: Optional[str] = field(
        default="token-abc123",
        metadata={"help": "The API key."},
    )

    @property
    def tokenizer(self) -> str:
        if self.override_tokenizer:
            return self.override_tokenizer
        return self.model


class OpenAIChatApiClientConfig(BaseApiClientConfig):
    additional_sampling_params: str = field(
        default="{}",
        metadata={
            "help": "Additional sampling params to send with the each request to the LLM API. "
            "By default, no additional sampling params are sent."
        },
    )

    def __post_init__(self):
        self.additional_sampling_params_dict = json.loads(
            self.additional_sampling_params
        )