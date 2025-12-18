from dataclasses import dataclass
from typing import Any, Dict

from veeksha.new.types import ChannelModality


@dataclass
class Request:
    """Configuration for a request to the LLM API.

    Args:
        id: Unique request ID.
        channels: Content of the request, indexed by modality.
    """

    id: int
    channels: Dict[ChannelModality, Any]  # content

    def __str__(self) -> str:
        return f"RequestConfig(id={self.id})"
