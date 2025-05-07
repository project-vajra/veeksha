from typing import Dict, Optional, Any

from veeksha.utils.dataclasses import frozen_dataclass


@frozen_dataclass
class Response:
    """The response object from the LLM API.

    Args:
        id: id of the response
        text: text from LLM
        logprobs: logprobs from LLM
    """

    id: int
    text: str
    logprobs: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"Response(id={self.id}, text={self.text}, logprobs={self.logprobs})"
