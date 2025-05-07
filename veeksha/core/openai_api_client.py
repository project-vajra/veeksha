import json
import os
import time
from typing import List, Optional, Tuple, Dict, Any

import requests

from veeksha.datatypes.request_config import RequestConfig
from veeksha.datatypes.response import Response
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics
from veeksha.config import ClientConfig
from veeksha.utils.hf_utils import get_tokenizer
from veeksha.enums import APIType

logger = init_logger(__name__)

# Maximum number of responses to store for token counting
MAX_RESPONSES_ALLOWED_TO_STORE = 5


class OpenAIAPIClient:
    """Client for OpenAI Chat Completions API."""

    def __init__(self, config: ClientConfig) -> None:
        assert config.api_type in [APIType.CHAT_COMPLETION, APIType.COMPLETION]
        assert config.api_url is not None

        self.model_name = config.model
        self.api_type = config.api_type

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            key = ""
            logger.warning(
                "Warning: OPENAI_API_KEY environment variable not set. Defaulting to empty string."
            )

        self.headers = {"Authorization": f"Bearer {key}"}
        self.address = self.construct_api_url(config.api_url, self.api_type)

        self.tokenizer = get_tokenizer(
            config.tokenizer or self.model_name,
            trust_remote_code=True,
        )

        self.start_time = time.monotonic()

    def construct_api_url(self, api_url: str, api_type: APIType) -> str:
        address = api_url

        if not address.endswith("/"):
            address = address + "/"

        if api_type == APIType.CHAT_COMPLETION:
            address += "chat/completions"
        elif api_type == APIType.COMPLETION:
            address += "completions"
        else:
            raise ValueError(f"Unknown API type: {api_type}")

        return address

    def get_token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

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

    def construct_body(self, request_config: RequestConfig) -> dict[str, str | list[dict[str, str]] | bool]:
        body: dict[str, str | list[dict[str, str]] | bool] = {
            "model": self.model_name,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

        if self.api_type == APIType.CHAT_COMPLETION:
            message = [
                {"role": "user", "content": request_config.prompt},
            ]
            body["messages"] = message
            body["stream"] = True
        elif self.api_type == APIType.COMPLETION:
            body["prompt"] = request_config.prompt
        else:
            raise ValueError(f"Unknown API type: {self.api_type}")

        return body

    def get_content(self, data: dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        if self.api_type == APIType.CHAT_COMPLETION:
            return data["choices"][0]["delta"].get("content", None), data["choices"][0]["delta"].get("logprobs", None)
        elif self.api_type == APIType.COMPLETION:
            return data["choices"][0].get("text", None), data["choices"][0].get("logprobs", None)
        else:
            raise ValueError(f"Unknown API type: {self.api_type}")

    def send_llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[RequestMetrics, Response]:
        body = self.construct_body(request_config)

        inter_token_times: List[float] = []
        error_msg: Optional[str] = None
        error_response_code: Optional[int] = None
        tokens_received: int = 0
        generated_text: str = ""
        logprobs: Dict[str, Any] = {}
        previous_responses: List[str] = []
        previous_token_count: int = 0

        most_recent_received_token_time: float = time.monotonic()
        request_dispatched_at: float = time.monotonic() - self.start_time

        try:
            with requests.post(
                self.address, json=body, timeout=None, headers=self.headers, stream=True
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
                        raise RuntimeError(data["error"]["message"])

                    chunk_content, chunk_logprobs = self.get_content(data)

                    if not chunk_content:
                        continue

                    (
                        current_tokens_received,
                        previous_token_count,
                    ) = self.get_current_tokens_received(
                        previous_responses=previous_responses,
                        current_response=chunk_content,
                        previous_token_count=previous_token_count,
                    )

                    tokens_received += current_tokens_received
                    inter_token_times.append(
                        time.monotonic() - most_recent_received_token_time
                    )
                    # Sometimes multiple tokens are sent in one chunk
                    if current_tokens_received > 1:
                        inter_token_times.extend(
                            [0] * (current_tokens_received - 1)
                        )
                    most_recent_received_token_time = time.monotonic()
                    generated_text += chunk_content

                    if chunk_logprobs:
                        logprobs["token_logprobs"].append(chunk_logprobs["token_logprobs"])
                        logprobs["top_logprobs"].append(chunk_logprobs["top_logprobs"])
        except Exception as e:
            logger.error(f"Warning Or Error: ({error_response_code}) {e}")

        metrics = RequestMetrics(
            num_prompt_tokens=request_config.num_prompt_tokens,
            num_output_tokens=tokens_received,
            request_dispatched_at=request_dispatched_at,
            inter_token_times=inter_token_times,
            error_code=error_response_code,
            error_msg=error_msg,
        )

        response = Response(
            id=request_config.id,
            text=generated_text,
            logprobs=logprobs,
        )

        return metrics, response
