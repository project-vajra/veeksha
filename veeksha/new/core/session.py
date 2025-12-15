from dataclasses import dataclass
from typing import Dict

from veeksha.new.core.request import Request
from veeksha.new.core.session_graph import SessionGraph


@dataclass
class Session:
    """A single Veeksha session.

    Args:
        id: Unique session ID
        session_graph: Session graph of the session (just structure, no content)
        requests: Requests in the session (actual content)
        cancel_session_on_failure: Whether to cancel the session on failure of any request
    """

    id: int
    session_graph: SessionGraph
    requests: Dict[int, Request]
    cancel_session_on_failure: bool = True
