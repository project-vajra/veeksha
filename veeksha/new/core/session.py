from typing import List

from pydantic import BaseModel

from veeksha.new.core.request import Request


# TODO: generalize to session graph (a DAG of requests)
class Session(BaseModel):
    """A single Veeksha session.

    Args:
        session_id: Unique session ID.
        session_total_requests: Total number of requests planned for the session.
        cancel_session_on_failure: Whether to cancel the session on failure.
        request_graph: List of requests in the session.
        session_start_time: Absolute timestamp for first-in-session dispatch (seconds)
    """

    session_id: int
    session_total_requests: int
    cancel_session_on_failure: bool = True

    request_graph: List[Request]

    session_start_time: float
