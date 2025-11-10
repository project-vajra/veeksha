from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class Request:
    """The configuration for a request to the LLM API.

    Args:
        model: The model to use.
        prompt: The prompt to provide to the LLM API.
        dispatch_delay: The delay in seconds before dispatching the request to the LLM API.
        sampling_params: Additional sampling parameters to send with the request.
            For more information see the Router app's documentation for the completions endpoint.
    """

    model: str
    prompt: Tuple[str, int]
    dispatch_delay: float
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    id: int = field(default_factory=lambda: Request.get_next_id())

    @staticmethod
    def get_next_id():
        if not hasattr(Request, "id_counter"):
            Request.id_counter = 0 # type: ignore

        Request.id_counter += 1 # type: ignore
        return Request.id_counter # type: ignore

    def __str__(self) -> str:
        return f"Request(id={self.id})"
