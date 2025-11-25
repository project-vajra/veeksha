from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel


class RequestConfig(BaseModel):
    """Configuration for a request to the LLM API.

    Args:
        model: The model to use.
        id: Unique request ID.
        prompt: Tuple of (prompt_text, token_count).
        session_id: Session identifier.
        session_sequence_index: Order within the session (0-based).
        session_total_requests: Total number of requests planned for the session.
        session_start_time: Absolute timestamp for first-in-session dispatch.
        wait_after_prev_response_s: Think time after previous response in session.
        sampling_params: LLM sampling parameters.
        llm_api: Target LLM API name.
    """

    # -- request metadata

    model: str
    id: int
    prompt: Tuple[str, int]
    sampling_params: Optional[Dict[str, Any]] = None
    llm_api: Optional[str] = None
    address_append_value: Optional[str] = None
    benchmark_id: str = "default"

    # -- session metadata
    session_id: int
    session_sequence_index: int
    session_total_requests: int
    cancel_session_on_failure: bool = True

    # -- scheduling

    # absolute scheduling for first-in-session requests
    # None for all other requests
    session_start_time: Optional[float] = None

    # None for first-in-session requests
    # delay since prev response of same session for all other requests
    wait_after_prev_response_s: Optional[float] = None

    def __str__(self) -> str:
        return f"RequestConfig(id={self.id})"
