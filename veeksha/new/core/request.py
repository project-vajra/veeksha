from dataclasses import dataclass
from typing import Any, Dict, Optional

from veeksha.new.types import ChannelModality


@dataclass
class Request:
    """Configuration for a request to the LLM API.

    Args:
        id: Unique request ID.
        channels: Content of the request, indexed by modality.
        model: The model to use.
        sampling_params: LLM sampling parameters.
        llm_api: Target LLM API name.
    """

    id: int
    channels: Dict[ChannelModality, Any]  # content

    # TODO: this should probably not be here
    model: str
    sampling_params: Optional[Dict[str, Any]] = None
    llm_api: Optional[str] = None
    address_append_value: Optional[str] = None

    def __str__(self) -> str:
        return f"RequestConfig(id={self.id})"
