"""Router client for OpenAI-compatible APIs.

This client supports *per-request* routing between:
- `/chat/completions` (streaming) for generation-style requests
- `/completions` (non-stream) for logprobs/loglikelihood-style requests

Routing is controlled by `request.metadata["api_mode"]`:
- "chat" (or unset) -> chat completions
- "completions" -> completions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from veeksha.new.client.base import BaseLLMClient
from veeksha.new.client.openai_chat import OpenAIChatCompletionsClient
from veeksha.new.client.openai_completions import OpenAICompletionsClient
from veeksha.new.core.request import Request
from veeksha.new.core.response import RequestResult
from veeksha.new.core.tokenizer import TokenizerProvider

if TYPE_CHECKING:
    from veeksha.new.config.client import OpenAIRouterClientConfig


class OpenAIRouterClient(BaseLLMClient):
    """Routes requests to chat or completions endpoints based on request metadata."""

    def __init__(
        self, config: OpenAIRouterClientConfig, tokenizer_provider: TokenizerProvider
    ) -> None:
        super().__init__(config)
        self._chat_client = OpenAIChatCompletionsClient(
            config=config, tokenizer_provider=tokenizer_provider
        )
        self._completions_client = OpenAICompletionsClient(
            config=config, tokenizer_provider=tokenizer_provider
        )

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
    ) -> RequestResult:
        if isinstance(request.metadata, dict) and request.metadata.get("api_mode") == (
            "completions"
        ):
            return await self._completions_client.send_request(
                request=request,
                session_id=session_id,
                session_total_requests=session_total_requests,
            )
        return await self._chat_client.send_request(
            request=request,
            session_id=session_id,
            session_total_requests=session_total_requests,
        )
