"""Response data structures for the new Veeksha framework.

These dataclasses bridge the gap between LLM client responses and what evaluators need.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from veeksha.new.types import ChannelModality


@dataclass
class ChannelResponse:
    """Response data for a single channel.

    Attributes:
        modality: The channel modality (TEXT, IMAGE, AUDIO, VIDEO)
        content: Modality-specific content (e.g., text string, image bytes)
        metrics: Channel-specific metrics (e.g., inter_token_times for TEXT)
    """

    modality: ChannelModality
    content: Any
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestResult:
    """Result from executing a single request.

    This is the output of an LLM client call, containing both timing information
    and per-channel response data that evaluators can consume.

    Attributes:
        request_id: Unique request identifier
        session_id: Session this request belongs to
        dispatched_at: Monotonic timestamp when request was dispatched
        completed_at: Monotonic timestamp when response was received
        channels: Per-channel response data
        success: True if request completed without error
        error_code: HTTP error code if request failed
        error_msg: Error message if request failed
    """

    request_id: int
    session_id: int

    dispatched_at: float
    completed_at: float

    # per-channel responses
    channels: Dict[ChannelModality, ChannelResponse] = field(default_factory=dict)

    success: bool = True
    error_code: Optional[int] = None
    error_msg: Optional[str] = None

    @property
    def latency(self) -> float:
        """Total request latency in seconds."""
        return self.completed_at - self.dispatched_at
