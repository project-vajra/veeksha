from dataclasses import dataclass
from typing import Dict

from veeksha.new.core.request_content import BaseChannelRequestContent
from veeksha.new.types import ChannelModality


@dataclass
class Request:
    """Configuration for a request to the LLM API.

    This object contains the input content for each modality (channel)
    and any per-request settings.
    """

    id: int
    channels: Dict[ChannelModality, BaseChannelRequestContent]

    def __str__(self) -> str:
        return f"RequestConfig(id={self.id})"
