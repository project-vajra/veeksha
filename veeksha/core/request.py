from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from veeksha.core.request_content import BaseChannelRequestContent
from veeksha.core.requested_output import RequestedOutputSpec
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
        requested_output: Specification for expected output from the model.
    """

    id: int
    channels: Dict[ChannelModality, BaseChannelRequestContent]
    history: List[Dict[str, Any]] = field(default_factory=list)
    session_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    requested_output: Optional[RequestedOutputSpec] = None

    def __str__(self) -> str:
        return f"RequestConfig(id={self.id})"
