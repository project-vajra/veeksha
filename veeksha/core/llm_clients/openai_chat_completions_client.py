import asyncio
import json
import os
import time
from typing import List, Optional, Tuple

import aiohttp
import requests

from veeksha.core.llm_clients.base_llm_client import BaseLLMClient
from veeksha.core.request_config import RequestConfig
from veeksha.core.response import Response
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)

# Maximum number of responses to store for token counting
MAX_RESPONSES_ALLOWED_TO_STORE = 5


class OpenAIChatCompletionsClient(BaseLLMClient):
    """Client for OpenAI Chat Completions API."""

    def __init__(self, model_name: str, tokenizer_name: str) -> None:
        super().__init__(model_name, tokenizer_name)
        self.address = os.environ.get("OPENAI_API_BASE")
        if not self.address:
            self.address = "http://localhost:8000/v1"
            logger.warning(
                "Warning: OPENAI_API_BASE environment variable not set. Defaulting to localhost."
            )
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            self.key = ""
            logger.warning(
                "Warning: OPENAI_API_KEY environment variable not set. Defaulting to empty string."
            )
        self.start_time = time.monotonic()

    def total_tokens(self, response_list: List[str]) -> int:
        merged_content = "".join(response_list)
        return self.get_token_length(merged_content)

    def get_current_tokens_received(
        self,
        previous_responses: List[str],
        current_response: str,
        previous_token_count: int,
    ) -> Tuple[int, int]:
        previous_responses.append(current_response)
        current_tokens_received = (
            self.total_tokens(previous_responses) - previous_token_count
        )
        if len(previous_responses) > MAX_RESPONSES_ALLOWED_TO_STORE:
            previous_responses.pop(0)
        previous_token_count = self.total_tokens(previous_responses)
        return current_tokens_received, previous_token_count

    async def send_llm_request(
        self, request_config: RequestConfig, request_dispatched_at: float
    ) -> Tuple[RequestMetrics, Response]:
        prompt, prompt_len = request_config.prompt

        message = [
            {"role": "user", "content": prompt},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

        headers = {"Authorization": f"Bearer {self.key}"}
        address = self.address

        if not address:
            raise ValueError("No host provided.")
        if not address.endswith("/"):
            address = address + "/"
        address += request_config.address_append_value or "chat/completions"

        inter_token_times = []
        error_msg = None
        error_response_code = None
        tokens_received = 0
        generated_text = ""
        previous_responses = []
        previous_token_count = 0

        most_recent_received_token_time = time.monotonic()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    address, json=body, headers=headers
                ) as response:
                    if response.status != 200:
                        error_response_code = response.status
                        error_msg = await response.text()
                        logger.error(f"Request Error: {error_msg}")
                        response.raise_for_status()
                    
                    # Process the stream response
                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()
                        
                        if not line_str:
                            continue
                            
                        if line_str.startswith('data: '):
                            line_str = line_str[len('data: '):]
                            
                        if line_str in ["[DONE]"]:
                            continue

                        try:
                            data = json.loads(line_str)
                        except json.JSONDecodeError:
                            logger.error(f"JSON decode error with line: {line_str}")
                            continue  # Skip malformed JSON

                        if "error" in data:
                            error_msg = data["error"]["message"]
                            error_response_code = data["error"]["code"]
                            raise RuntimeError(data["error"]["message"])

                        delta = data["choices"][0]["delta"]
                        if delta.get("content", None):
                            (
                                current_tokens_received,
                                previous_token_count,
                            ) = self.get_current_tokens_received(
                                previous_responses=previous_responses,
                                current_response=delta["content"],
                                previous_token_count=previous_token_count,
                            )

                            tokens_received += current_tokens_received
                            inter_token_times.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                            if current_tokens_received > 1:
                                inter_token_times.extend(
                                    [0] * (current_tokens_received - 1)
                                )
                            most_recent_received_token_time = time.monotonic()
                            generated_text += delta["content"]
        except Exception as e:
            logger.error(f"Warning Or Error: ({error_response_code}) {e}")

        metrics = RequestMetrics(
            request_dispatched_at=request_dispatched_at,
            inter_token_times=inter_token_times,
            num_prompt_tokens=prompt_len,
            num_output_tokens=tokens_received,
            error_code=error_response_code,
            error_msg=error_msg,
        )

        response = Response(
            id=request_config.id,
            text=generated_text,
        )

        return metrics, response
