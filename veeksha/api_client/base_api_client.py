from abc import ABC, abstractmethod
from typing import Optional, Tuple

import aiohttp

from veeksha.core.hf_utils import get_tokenizer
from veeksha.core.request import Request
from veeksha.core.response import Response
from veeksha.metrics.request_metrics import RequestMetrics
from veeksha.config.api_client_config import BaseApiClientConfig



class BaseApiClient(ABC):
    """A client for making requests to a LLM API Endpoint."""

    def __init__(self, config: BaseApiClientConfig) -> None:
        self.config = config
        self.model_name = config.model
        self.tokenizer = get_tokenizer(
            config.tokenizer,
            trust_remote_code=True,
        )

    def get_token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @abstractmethod
    async def send_request(
        self, request: Request, session: aiohttp.ClientSession
    ) -> Tuple[RequestMetrics, Optional[Response]]:
        """Make a single request to a LLM API

        Returns:
            Metrics about the performance characteristics of the request.
            The response from the LLM API.
        """
        ...
