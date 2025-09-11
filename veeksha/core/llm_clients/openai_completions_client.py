import json
import os
import time
from typing import List, Tuple

import requests

from veeksha.core.llm_clients.base_llm_client import BaseLLMClient
from veeksha.core.request_config import RequestConfig
from veeksha.core.response import Response
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)

# Maximum number of responses to store for token counting
MAX_RESPONSES_ALLOWED_TO_STORE = 5


class OpenAICompletionsClient(BaseLLMClient):
    """Client for OpenAI Completions API."""

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

    def send_llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[RequestMetrics, Response]:
        # The request_config.prompt is expected to be a tuple: (prompt_text, prompt_length)
        prompt, prompt_len = request_config.prompt

        # Completions API should only be used with lm_eval loglikelihood tasks.
        model = request_config.model
        body = {
            "model": model,
            "prompt": prompt,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

        headers = {"Authorization": f"Bearer {self.key}"}
        address = self.address

        if not address:
            raise ValueError("No host provided.")
        if not address.endswith("/"):
            address = address + "/"
        # Change the endpoint from "chat/completions" to "completions"
        address += request_config.address_append_value or "completions"

        inter_token_times = []
        error_msg = None
        error_response_code = None
        tokens_received = 0
        generated_text = ""
        logprobs = {}
        previous_responses = []
        previous_token_count = 0

        most_recent_received_token_time = time.monotonic()
        request_dispatched_at = time.monotonic() - self.start_time

        try:
            with requests.post(
                address, json=body, timeout=None, headers=headers, stream=False
            ) as response:
                if response.status_code != 200:
                    error_response_code = response.status_code
                    error_msg = response.text
                    logger.error(f"Request Error: {error_msg}")
                    response.raise_for_status()

                for chunk in response.iter_lines(chunk_size=None):
                    chunk = chunk.strip()
                    if not chunk:
                        continue

                    # Remove the "data: " prefix if present.
                    stem = "data: "
                    if chunk.startswith(stem.encode()):
                        chunk = chunk[len(stem) :]

                    if chunk in [b"[DONE]", "[DONE]"]:
                        continue

                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        logger.error(f"JSON decode error with chunk: {chunk}")
                        continue  # Skip malformed JSON

                    if "error" in data:
                        error_msg = data["error"]["message"]
                        error_response_code = data["error"].get("code", None)
                        raise RuntimeError(error_msg)

                    text_chunk = data["choices"][0].get("text", "")
                    if text_chunk:
                        current_tokens_received, previous_token_count = (
                            self.get_current_tokens_received(
                                previous_responses=previous_responses,
                                current_response=text_chunk,
                                previous_token_count=previous_token_count,
                            )
                        )
                        tokens_received += current_tokens_received
                        inter_token_times.append(  # Just get TTFT
                            time.monotonic() - most_recent_received_token_time
                        )
                        most_recent_received_token_time = time.monotonic()
                        generated_text += text_chunk
                        if "logprobs" in data["choices"][0]:
                            logprobs = data["choices"][0]["logprobs"]
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
            logprobs=logprobs,
        )

        return metrics, response
