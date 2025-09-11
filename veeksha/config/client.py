import json
from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.core.llm_clients import SUPPORTED_APIS


@frozen_dataclass(allow_from_file=True)
class ClientConfig:
    model: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "The model to use for this load test."},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for this load test. By default, the tokenizer is inferred from the model."
        },
    )
    num_clients: int = field(
        default=2,
        metadata={"help": "The number of clients to use for benchmark."},
    )
    num_concurrent_requests_per_client: int = field(
        default=5,
        metadata={"help": "The number of concurrent requests to send per client."},
    )
    additional_sampling_params: str = field(
        default="{}",
        metadata={
            "help": "Additional sampling params to send with the each request to the LLM API. "
            "By default, no additional sampling params are sent."
        },
    )
    llm_api: str = field(
        default="openai_chat",
        metadata={
            "help": f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
        },
    )
    address_append_value: str = field(
        default="chat/completions",
        metadata={"help": "The address append value for OpenAI API."},
    )
    request_timeout: int = field(
        default=60,
        metadata={
            "help": "The timeout for each request to the LLM API in seconds."
        },
    )

    def __post_init__(self):
        self.additional_sampling_params_dict = {}

        if self.additional_sampling_params:
            self.additional_sampling_params_dict = json.loads(
                self.additional_sampling_params
            )

        if self.tokenizer is None:
            self.tokenizer = self.model
