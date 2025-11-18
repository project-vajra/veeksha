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
        delay: Relative scheduling delay (inter-arrival if index=0, think-time if >0).
        arrival_time: Absolute timestamp override (e.g. for trace replay).
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
    cancel_session_on_failure: bool = True

    # -- scheduling

    # if seq_index == 0: delay since prev session start
    # if seq_index > 0: delay since prev response
    delay: float = 0.0

    # absolute scheduling (overrides delay)
    arrival_time: Optional[float] = None

    def __str__(self) -> str:
        return f"RequestConfig(id={self.id})"
