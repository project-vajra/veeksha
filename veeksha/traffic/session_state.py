"""State management for scheduled sessions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from veeksha.core.request import Request
from veeksha.core.session import Session


@dataclass
class ScheduledSessionState:
    """State for a single scheduled session.

    Args:
        session: The session object being tracked
        session_start_time: When the session should start (relative to scheduler epoch)
        completions: Mapping of node_id to completion_time for completed nodes
        pending_nodes: Set of node IDs that haven't been queued to the ready queue yet
        queued_nodes: Set of node IDs that have been queued but not yet completed
        node_histories: Mapping of node_id to list of node history entries
        is_canceled: Whether this session has been canceled
        cancel_on_failure: Whether to cancel remaining nodes on any failure
    """

    session: Session
    session_start_time: float
    completions: Dict[int, float] = field(default_factory=dict)
    pending_nodes: Set[int] = field(default_factory=set)
    queued_nodes: Set[int] = field(default_factory=set)
    node_histories: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    is_canceled: bool = False
    cancel_on_failure: bool = True


@dataclass(order=True)
class ScheduledItem:
    """A request scheduled for dispatch at a specific time."""

    # ordered by ready at
    ready_at: float

    # not ordered
    request_id: int = field(compare=False)
    request: Request = field(compare=False)
