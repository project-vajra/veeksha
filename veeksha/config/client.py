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
        metadata={"help": "The default number of clients to use for benchmark."},
    )
    num_concurrent_requests_per_client: int = field(
        default=5,
        metadata={"help": "The number of concurrent requests each client can manage."},
    )
    auto_spawn_new_clients: bool = field(
        default=False,
        metadata={
            "help": "If True, spawn additional clients when requests are ready to be dispatched but all clients are busy. Spawned clients use the same per-client concurrency.",
        },
    )
    max_clients: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum total number of clients allowed when auto-spawn is enabled. None means unlimited.",
        },
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
        metadata={"help": "The timeout for each request to the LLM API (in seconds)."},
    )
    min_tokens_param: Optional[str] = field(
        default="min_tokens",
        metadata={
            "help": "Name of server parameter for minimum tokens to be generated. If omitted or non accepted by the server, fallback to prompt-based minimum token request (append instruction to prompt to generate at least the requested number of tokens)."
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

        if self.num_clients < 1:
            raise ValueError("num_clients must be greater than 0")
        if self.num_concurrent_requests_per_client < 1:
            raise ValueError(
                "num_concurrent_requests_per_client must be greater than 0"
            )
        if (
            self.auto_spawn_new_clients
            and self.max_clients is not None
            and self.max_clients < 1
        ):
            raise ValueError(
                "max_clients must be greater than 0 when auto_spawn_new_clients is set"
            )
        if (
            self.auto_spawn_new_clients
            and self.max_clients is not None
            and self.num_clients >= self.max_clients
        ):
            raise ValueError(
                "num_clients must be less than max_clients when auto_spawn_new_clients is True"
            )
