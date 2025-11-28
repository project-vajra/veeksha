import asyncio
import os
import time
import uuid
from typing import Dict, Optional, Tuple

import aiohttp  # type: ignore

from revati.client import ClientType  # type: ignore
from revati.client.helper import create_thread_local_revati_client, get_virtual_time, is_revati_enabled

from veeksha.core.llm_clients.base_llm_client import BaseLLMClient
from veeksha.core.llm_clients.streaming_mixin import StreamingMixin
from veeksha.core.request_config import RequestConfig
from veeksha.core.response import Response
from veeksha.dashboard.events import TokenBatchEvent
from veeksha.dashboard.handler import emit_dashboard_event
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics


logger = init_logger(__name__)


class OpenAIChatCompletionsClient(BaseLLMClient, StreamingMixin):
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

        if is_revati_enabled():
            create_thread_local_revati_client(f"veeksha-client-{str(uuid.uuid4())[:8]}", ClientType.OBSERVER)

        self.start_time = get_virtual_time()

    def _update_metrics_from_chunk(
        self,
        data: Dict,
        inter_token_times: list,
        previous_responses: list,
        previous_token_count: int,
        most_recent_received_token_time: float,
    ) -> Tuple[int, int, str, float]:
        """Update metrics and generated text from a single data chunk."""
        generated_text_chunk = ""
        tokens_received_chunk = 0

        delta = data.get("choices", [{}])[0].get("delta", {})
        if content_chunk := delta.get("content"):
            (
                current_tokens_received,
                previous_token_count,
            ) = self.get_current_tokens_received(
                previous_responses=previous_responses,
                current_response=content_chunk,
                previous_token_count=previous_token_count,
            )

            tokens_received_chunk += current_tokens_received
            inter_token_times.append(get_virtual_time() - most_recent_received_token_time)
            if current_tokens_received > 1:
                inter_token_times.extend([0] * (current_tokens_received - 1))

            most_recent_received_token_time = get_virtual_time()
            generated_text_chunk = content_chunk

        return (
            tokens_received_chunk,
            previous_token_count,
            generated_text_chunk,
            most_recent_received_token_time,
        )

    async def send_llm_request(
        self, request_config: RequestConfig, session: aiohttp.ClientSession
    ) -> Tuple[RequestMetrics, Optional[Response]]:
        prompt, prompt_len = request_config.prompt

        message = [
            {"role": "user", "content": prompt},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
            "return_token_ids": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
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

        # In case of mismatched sleep-jump distributions, change this to time.time() to eliminate input distribution considerations
        request_dispatched_at = get_virtual_time()

        # Respect a local cap on tokens to avoid mismatches with server/tokenizer
        max_tokens_limit = None
        if isinstance(request_config.sampling_params, dict):
            max_tokens_limit = request_config.sampling_params.get("max_tokens")

        last_token_batch_emission = 0
        is_first_emission = True  # Track if this is the first dashboard event emission

        try:
            async with session.post(address, json=body, headers=headers) as response:
                response.raise_for_status()

                # Initialize the most recent received token time to the time the request was dispatched
                # This is done to remove the request dispatch and accept time from ttft calculation
                most_recent_received_token_time = get_virtual_time()

                async for data in self._process_stream(response):
                    tokens_received_chunk = 0  # Track tokens received in this chunk
                    if "error" in data:
                        err = data.get("error") or {}
                        error_msg = err.get("message", "Unknown error")
                        code_value = err.get("code")
                        error_response_code = (
                            code_value if isinstance(code_value, int) else 400
                        )
                        break  # Stop processing on error
                    delta = data["choices"][0]["delta"]

                    # If output doesn't have token_ids, skip this chunk
                    # TODO: this only works for vllm right now, we need to generalize this
                    if "token_ids" not in data["choices"][0]:
                        continue

                    # Skip 1st [chunk with role] and last [finish_reason : length] chunks with no tokens
                    if not data["choices"][0].get("token_ids"):
                        continue

                    chunk_arrival_monotonic = (
                        get_virtual_time()
                    )
                    current_tokens_received = len(data["choices"][0]["token_ids"])

                    allowable_to_add = current_tokens_received
                    if allowable_to_add > 0:
                        tokens_received_chunk += allowable_to_add
                        inter_token_times.append(
                            chunk_arrival_monotonic
                            - most_recent_received_token_time
                        )
                        if allowable_to_add > 1:
                            inter_token_times.extend([0] * (allowable_to_add - 1))
                        tokens_received += allowable_to_add
                        most_recent_received_token_time = chunk_arrival_monotonic

                        generated_text += delta["content"]

                        # Truncate generated_text to exactly tokens_received tokens
                        if isinstance(max_tokens_limit, int):
                            output_token_ids = self.tokenizer.encode(generated_text)
                            if len(output_token_ids) > tokens_received:
                                generated_text = self.tokenizer.decode(
                                    output_token_ids[:tokens_received]
                                )

                    # If we've reached or exceeded the cap, throw an error
                    if (
                        isinstance(max_tokens_limit, int)
                        and tokens_received > max_tokens_limit
                    ):
                        logger.exception(
                            f"Tokens received ({tokens_received}) exceeded max tokens limit ({max_tokens_limit})"
                        )
                        error_msg = f"Tokens received ({tokens_received}) exceeded max tokens limit ({max_tokens_limit})"
                        error_response_code = 400
                        break

                    if (
                        tokens_received_chunk > 0
                        and (tokens_received - last_token_batch_emission) >= 10
                    ):  # TODO: make this configurable/a constant
                        current_ttft_ms = (
                            inter_token_times[0] * 1000 if inter_token_times else None
                        )
                        current_tpot_ms = None
                        recent_tbt_ms = None
                        if len(inter_token_times) > 1:
                            from statistics import mean

                            current_tpot_ms = mean(inter_token_times[1:]) * 1000
                            # Get recent TBT values (last 10 or all available)
                            recent_tbt_ms = [
                                t * 1000 for t in inter_token_times[1:][-10:]
                            ]

                        if request_config.id is not None:
                            emit_dashboard_event(
                                TokenBatchEvent(
                                    request_id=request_config.id,
                                    timestamp=time.time(),
                                    tokens_received_this_batch=tokens_received_chunk,
                                    total_output_tokens=tokens_received,
                                    ttft_ms=current_ttft_ms,
                                    current_tpot_ms=current_tpot_ms,
                                    is_first_token=is_first_emission,
                                    recent_tbt_ms=recent_tbt_ms,
                                    benchmark_id=request_config.benchmark_id,
                                )
                            )
                        last_token_batch_emission = tokens_received
                        is_first_emission = (
                            False  # After first emission, no longer first
                        )

                if (
                    isinstance(request_config.sampling_params, dict)
                    and request_config.sampling_params.get("min_tokens") is not None
                ):
                    min_tokens = request_config.sampling_params.get("min_tokens")
                    if isinstance(min_tokens, int) and tokens_received < min_tokens:
                        logger.exception(
                            f"Tokens received ({tokens_received}) is less than min tokens ({min_tokens})"
                        )
                        error_msg = f"Tokens received ({tokens_received}) is less than min tokens ({min_tokens})"
                        error_response_code = 400

        except aiohttp.ClientResponseError as e:
            error_response_code = e.status
            error_msg = error_msg or (e.message if hasattr(e, "message") else str(e))
            logger.warning(f"HTTP Error: status={error_response_code} msg={error_msg}")
        except aiohttp.ClientConnectorError as e:
            error_response_code = 503
            error_msg = error_msg or str(e)
            logger.warning(f"Connection Error: ({error_response_code}) {error_msg}")
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            error_response_code = 408
            error_msg = error_msg or "Request timed out"
            logger.warning(f"Timeout Error: ({error_response_code}) {error_msg}")
        except Exception as e:
            error_response_code = error_response_code or 520
            error_msg = error_msg or str(e)
            logger.exception(
                f"An unexpected error occurred: ({error_response_code}) {error_msg}"
            )

        metrics = RequestMetrics(
            request_id=request_config.id,
            request_dispatched_at=request_dispatched_at,
            inter_token_times=inter_token_times,
            num_prompt_tokens=prompt_len,
            num_output_tokens=tokens_received,
            error_code=error_response_code,
            error_msg=error_msg,
            benchmark_id=request_config.benchmark_id,
        )

        generated_response: Optional[Response]
        if error_msg or error_response_code:
            generated_response = None
        else:
            generated_response = Response(
                id=request_config.id,
                text=generated_text,
            )

        return metrics, generated_response
