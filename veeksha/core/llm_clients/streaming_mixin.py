import json
from typing import AsyncIterator, Dict, List, Protocol, Tuple, cast

import httpx

from veeksha.logger import init_logger

logger = init_logger(__name__)


class _HasTokenLength(Protocol):
    def get_token_length(self, text: str) -> int: ...


class StreamingMixin:
    """Shared streaming and token-count helpers for async LLM clients.

    Requires the consumer class to implement `get_token_length(text: str) -> int`.
    """

    MAX_RESPONSES_ALLOWED_TO_STORE: int = 5

    async def _process_stream(self, response: httpx.Response) -> AsyncIterator[Dict]:
        """Process Server-Sent Events (SSE) stream and yield parsed JSON dicts.

        Skips non-data lines and stops on "[DONE]" sentinel.
        """
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    logger.exception(f"JSON decode error with chunk: {data_str}")

    def total_tokens(self, response_list: List[str]) -> int:
        """Return token length of concatenated responses using client's tokenizer."""
        merged_content = "".join(response_list)
        provider = cast(_HasTokenLength, self)
        return provider.get_token_length(merged_content)

    def get_current_tokens_received(
        self,
        previous_responses: List[str],
        current_response: str,
        previous_token_count: int,
    ) -> Tuple[int, int]:
        """Update rolling token counters for streamed text.

        Returns a tuple of (new_tokens_in_this_chunk, updated_previous_token_count).
        """
        previous_responses.append(current_response)
        current_tokens_received = (
            self.total_tokens(previous_responses) - previous_token_count
        )
        if len(previous_responses) > self.MAX_RESPONSES_ALLOWED_TO_STORE:
            previous_responses.pop(0)
        previous_token_count = self.total_tokens(previous_responses)
        return current_tokens_received, previous_token_count
