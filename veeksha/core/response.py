from typing import Dict, List, Optional

from pydantic import BaseModel


class Response(BaseModel):
    """The response object from the LLM API.

    Args:
        id: id of the response
        text: text from LLM
        logprobs: logprobs from LLM
    """

    id: Optional[int] = None
    text: str
    logprobs: Optional[List[Dict]] = None

    def __str__(self) -> str:
        return f"Response(id={self.id}, text={self.text}, logprobs={self.logprobs})"
