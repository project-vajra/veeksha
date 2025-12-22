from dataclasses import dataclass, field
from typing import Any, Dict, List

from veeksha.new.core.request_content import BaseChannelRequestContent
from veeksha.new.types import ChannelModality


@dataclass
class Request:
    """Configuration for a request to the LLM API.

    This object contains the input content for each modality (channel).
    """

    id: int
    channels: Dict[ChannelModality, BaseChannelRequestContent]
    history: List[Dict[str, Any]] = field(default_factory=list)
    # node id, parent nodes, wait after ready, history parent. Useful for saving to trace
    session_context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"RequestConfig(id={self.id})"
