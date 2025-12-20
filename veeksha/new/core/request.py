from dataclasses import dataclass, field
from typing import Any, Dict, List

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
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        return f"RequestConfig(id={self.id})"
