from typing import Any, Dict

from veeksha.utils.dataclasses import frozen_dataclass


@frozen_dataclass
class RequestConfig:
    """The configuration for a request to the LLM API.

    Args:
        id: The ID of the request.
        prompt: The prompt to provide to the LLM API.
        num_prompt_tokens: The number of tokens in the prompt.
        sampling_params: Additional sampling parameters to send with the request.
            For more information see the Router app's documentation for the completions
    """

    id: int
    prompt: str
    num_prompt_tokens: int
    sampling_params: Dict[str, Any]

    def __str__(self) -> str:
        return f"RequestConfig(id={self.id}, num_prompt_tokens={self.num_prompt_tokens}, sampling_params={self.sampling_params})"
