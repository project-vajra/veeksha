"""State management for individual sessions."""

from typing import Dict, Optional

from veeksha.core.request_config import RequestConfig


class SessionState:
    """Holds all state for a single session."""

    def __init__(self):
        # Requests waiting for their predecessor to complete, keyed by sequence index
        self.pending_requests: Dict[int, RequestConfig] = {}
        # Highest completed sequence index (-1 means no requests completed yet)
        self.completed_sequence: int = -1
        # Time when the last request in this session completed
        self.last_completion_time: Optional[float] = None
        # Whether this session has been canceled
        self.is_canceled: bool = False
        # Cancel-on-failure policy for this session
        self.cancel_on_failure: Optional[bool] = None
