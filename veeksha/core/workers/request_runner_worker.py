"""Async worker that processes requests using uvloop."""

import asyncio
import json
import os
import threading
from queue import Full, Queue
from typing import Any, Optional
import time

import uvloop

from veeksha.config.client import ClientConfig
from veeksha.core.context import WorkerContext
from veeksha.core.llm_clients import construct_client
from veeksha.core.response import Response
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


class InputOutputWriter:
    """Writes input prompts and generated outputs to a JSONL file in streaming fashion."""

    def __init__(self, output_file: str, enabled: bool = True):
        """Initialize the writer.

        Args:
            output_file: Path to the output JSONL file
            enabled: Whether writing is enabled
        """
        self.output_file = output_file
        self.enabled = enabled
        self.file_handle = None
        self.lock = threading.Lock()

        if self.enabled:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            self.file_handle = open(output_file, "w", encoding="utf-8")
            logger.info(f"Input/output data will be written to: {output_file}")

    def write_input_output(
        self,
        request_id: Optional[int],
        session_id: Optional[int],
        session_sequence_index: Optional[int],
        prompt: str,
        generated_text: Optional[str],
        chat_history: Optional[list] = None,
    ) -> None:
        """Write input prompt and generated output to the file.

        Args:
            request_id: The request ID
            session_id: The session ID
            session_sequence_index: The position within the session
            prompt: The input prompt text
            generated_text: The generated output text (None on error)
            chat_history: The chat history for this session (None if no session)
        """
        if not self.enabled or self.file_handle is None:
            return

        io_data = {
            "request_id": request_id,
            "session_id": session_id,
            "session_sequence_index": session_sequence_index,
            "input_prompt": prompt,
            "generated_text": generated_text,
            "chat_history": chat_history if chat_history else None,
        }

        # Write to file with lock for thread safety
        with self.lock:
            self.file_handle.write(json.dumps(io_data) + "\n")
            self.file_handle.flush()  # Ensure immediate write

    def close(self) -> None:
        """Close the file handle."""
        if self.file_handle is not None:
            with self.lock:
                self.file_handle.close()
                self.file_handle = None
            logger.info(f"Closed input/output file: {self.output_file}")


class RequestRunnerWorker:
    """Single async worker that processes requests using uvloop."""

    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        worker_context: WorkerContext,
        client_config: ClientConfig,
        chat_history,
        input_output_writer: Optional[InputOutputWriter] = None,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.worker_context = worker_context
        self.client_config = client_config
        self.input_output_writer = input_output_writer
        self._chat_history = chat_history

    def run(self) -> None:
        """Main worker loop - runs uvloop event loop for concurrent request processing.

        With GIL-free Python (python -Xgil=0), this thread can run in parallel with
        other threads while using uvloop for concurrent HTTP requests.
        """
        logger.info(f"Request runner worker {self.worker_context.worker_id} starting")

        # Install uvloop for this thread
        uvloop.install()

        # Run the async event loop
        asyncio.run(self._worker_loop())

        logger.info(f"Request runner worker {self.worker_context.worker_id} exiting")

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
            if request_config.session_id and len(self._chat_history[request_config.session_id]) != request_config.session_sequence_index*2:
                raise ValueError(f"Chat history length is incorrect. request={request_config}, chat_history={self._chat_history[request_config.session_id]}")

            metrics, response = await llm_client.send_llm_request(
                request_config,
                self.client_config.request_timeout,
                chat_history=self._chat_history[request_config.session_id],
            )
            completed_at = time.monotonic()

            # Write input/output data if enabled
            if self.input_output_writer is not None:
                prompt_text = request_config.prompt[0]  # Extract prompt text from tuple
                generated_text = response.text if response is not None else None

                # Get chat history before it's updated
                chat_history = (
                    self._chat_history[request_config.session_id]
                    if request_config.session_id
                    else None
                )

                self.input_output_writer.write_input_output(
                    request_id=request_config.id,
                    session_id=request_config.session_id,
                    session_sequence_index=request_config.session_sequence_index,
                    prompt=prompt_text,
                    generated_text=generated_text,
                    chat_history=chat_history,
                )

            # Update chat history
            if request_config.session_id:
                self._chat_history[request_config.session_id].append(
                    {
                        "role": "user",
                        "content": request_config.prompt[0],
                    }
                )
                if response:
                    self._chat_history[request_config.session_id].append(
                        {
                            "role": "assistant",
                            "content": response.text,
                        }
                    )

            await asyncio.to_thread(
                self.output_queue.put, (metrics, response, completed_at)
            )
        except asyncio.CancelledError:
            # task cancelled due to shutdown / timeout
            await self._emit_error_result(
                exception=None, request_config=request_config, cancelled=True
            )
            raise
        except Exception as e:
            logger.exception(
                "send_llm_request failed for async worker_id=%s",
                self.worker_context.worker_id,
            )
            await self._emit_error_result(exception=e, request_config=request_config)
        finally:
            self.worker_context.decrement_load()

    async def _emit_error_result(
        self,
        exception: Optional[BaseException],
        request_config: Optional[Any],
        cancelled: bool = False,
    ) -> None:
        """Emit an error RequestMetrics tuple to the output queue."""
        try:
            prompt_len = request_config.prompt[1] if request_config else 0
            request_id = request_config.id if request_config else None
            error_code = None
            error_msg = None
            completed_at = time.monotonic()
            if cancelled:
                error_msg = "Cancelled by Veeksha"
            elif exception is not None:
                error_code = getattr(
                    getattr(exception, "response", None), "status_code", None
                )
                error_msg = str(exception)

            metrics = RequestMetrics(
                request_dispatched_at=0.0,
                inter_token_times=[],
                num_prompt_tokens=prompt_len,
                num_output_tokens=0,
                error_msg=error_msg,
                error_code=error_code,
                request_id=request_id,
                benchmark_id=(
                    request_config.benchmark_id if request_config else "default"
                ),
                cancelled=cancelled,
            )
            result = (metrics, None, completed_at)
            try:
                self.output_queue.put_nowait(result)
            except Full:
                await asyncio.to_thread(self.output_queue.put, result)
        except Exception:
            logger.exception(
                "Failed to enqueue error result for worker %s",
                self.worker_context.worker_id,
            )