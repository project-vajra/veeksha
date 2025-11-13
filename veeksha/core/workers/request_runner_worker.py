"""Async worker that processes requests using uvloop."""

import asyncio
from queue import Queue
from typing import Any, Optional

import uvloop

from veeksha.config.client import ClientConfig
from veeksha.core.context import WorkerContext
from veeksha.core.llm_clients import construct_client
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


class RequestRunnerWorker:
    """Single async worker that processes requests using uvloop."""

    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        worker_context: WorkerContext,
        client_config: ClientConfig,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.worker_context = worker_context
        self.client_config = client_config

    def run(self) -> None:
        """Main worker loop - runs uvloop event loop for concurrent request processing.

        With GIL-free Python (python -Xgil=0), this thread can run in parallel with
        other threads while using uvloop for concurrent HTTP requests.
        """
        logger.debug("Request runner worker %s starting", self.worker_context.worker_id)

        # Install uvloop for this thread
        uvloop.install()

        # Run the async event loop
        asyncio.run(self._worker_loop())

        logger.debug("Request runner worker %s exiting", self.worker_context.worker_id)

    async def _worker_loop(self) -> None:
        """Main async event loop that dispatches requests as concurrent tasks."""
        # Construct LLM client
        llm_client = construct_client(
            model_name=self.client_config.model,
            tokenizer_name=self.client_config.tokenizer or self.client_config.model,
            llm_api=self.client_config.llm_api,
        )

        while True:
            # Use to_thread to avoid blocking the event loop on queue.get()
            request_config = await asyncio.to_thread(self.input_queue.get)

            if request_config is None:
                # Sentinel value to signal shutdown
                break

            # Increment load before creating task
            self.worker_context.increment_load()

            # Create async task for concurrent execution
            asyncio.create_task(self._process_request(llm_client, request_config))

    async def _process_request(self, llm_client, request_config: Any) -> None:
        """Process a single request asynchronously."""
        try:
            result = await llm_client.send_llm_request(
                request_config, self.client_config.request_timeout
            )
            self.output_queue.put(result)
        except Exception as e:
            logger.exception(
                "send_llm_request failed for async worker_id=%s",
                self.worker_context.worker_id,
            )
            self._emit_error_result(e, request_config)
        finally:
            self.worker_context.decrement_load()

    def _emit_error_result(self, e: Exception, request_config: Optional[Any]) -> None:
        """Emit an error RequestMetrics tuple to the output queue."""
        try:
            prompt_len = request_config.prompt[1] if request_config else 0
            request_id = request_config.id if request_config else None
            error_code = getattr(getattr(e, "response", None), "status_code", None)

            metrics = RequestMetrics(
                request_dispatched_at=0.0,
                inter_token_times=[],
                num_prompt_tokens=prompt_len,
                num_output_tokens=0,
                error_msg=str(e),
                error_code=error_code,
                request_id=request_id,
                benchmark_id=(
                    request_config.benchmark_id if request_config else "default"
                ),
            )
            self.output_queue.put((metrics, None))
        except Exception:
            logger.exception(
                "Failed to enqueue error result for worker %s",
                self.worker_context.worker_id,
            )
