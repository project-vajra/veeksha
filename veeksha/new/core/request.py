from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from veeksha.new.config.generator.channel import BaseChannelGeneratorConfig


class Request(BaseModel):
    """Configuration for a request to the LLM API.

    Args:
        model: The model to use.
        id: Unique request ID.
        prompt: Tuple of (prompt_text, token_count).
        session_id: Session identifier.
        session_sequence_index: Order within the session (0-based).
        session_total_requests: Total number of requests planned for the session.
        session_start_time: Absolute timestamp for first-in-session dispatch.
        wait_after_prev_response: Delay since previous response in session is completed (seconds).
        sampling_params: LLM sampling parameters.
        llm_api: Target LLM API name.
    """

    id: int
    # TODO: we might not need this if we only deal with session objects
    session_id: int

    # TODO: this assumes a linear structure of the session. A dependency structure should be embedded within each session
    session_sequence_index: int

    channels: List[BaseChannelGeneratorConfig]  # content

    # TODO: this should probably not be here
    model: str
    sampling_params: Optional[Dict[str, Any]] = None
    llm_api: Optional[str] = None
    address_append_value: Optional[str] = None

    # -- scheduling
    # delay since request is ready to be dispatched. Under a linear structure, this is the wait after the previous
    # request in the session is completed. For first-in-session requests, this is always 0.
    wait_after_ready: float

    def __str__(self) -> str:
        return f"RequestConfig(id={self.id})"
