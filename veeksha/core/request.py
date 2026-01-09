from dataclasses import dataclass, field
from typing import Any, Dict, List

from veeksha.core.request_content import BaseChannelRequestContent
from veeksha.types import ChannelModality


@dataclass
class Request:
    """Contains the input content for each modality (channel).

    Attributes:
        id: Unique (benchmark-scoped) request identifier
        channels: Input content for each modality (channel)
        history: History of the content of the request
        session_context: Session context. Useful for saving to trace.
        metadata: Metadata for the request, like per-request sampling params.
    """

    id: int
    channels: Dict[ChannelModality, BaseChannelRequestContent]
    history: List[Dict[str, Any]] = field(default_factory=list)
    session_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"RequestConfig(id={self.id})"
