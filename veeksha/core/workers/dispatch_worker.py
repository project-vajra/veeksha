"""Worker that schedules and dispatches requests to worker queue."""

import threading
import os
import json
from queue import Empty, Queue

from veeksha.core.context import BenchmarkContext, WorkerContext
from veeksha.core.dispatch_scheduler import DispatchScheduler
from veeksha.core.dispatcher import RequestDispatcher
from veeksha.core.requests_launcher import RequestsLauncher
from veeksha.core.workers.constants import QUEUE_GET_TIMEOUT_S
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics

logger = init_logger(__name__)


class DispatchedRequestWriter:
    """Writes dispatched request metadata to a JSONL file in streaming fashion."""

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
            logger.info(f"Dispatched requests will be written to: {output_file}")

    def write_request(self, request_config, dispatch_timestamp: float) -> None:
        """Write a request's metadata to the file.

        Args:
            request_config: The RequestConfig object being dispatched
            dispatch_timestamp: The timestamp when the request was dispatched
        """
        if not self.enabled or self.file_handle is None:
            return

        # Extract metadata from RequestConfig
        _, prompt_length = request_config.prompt

        request_data = {
            "request_id": request_config.id,
            "session_id": request_config.session_id,
            "session_sequence_index": request_config.session_sequence_index,
            "dispatch_timestamp": dispatch_timestamp,
            "dispatch_delay": request_config.dispatch_delay,
            "anchor_at_s": request_config.anchor_at_s,
            "wait_after_prev_response_s": request_config.wait_after_prev_response_s,
            "input_length": prompt_length,
            "output_length": request_config.sampling_params.get("max_tokens") if request_config.sampling_params else None,
            "model": request_config.model,
            "llm_api": request_config.llm_api,
            "benchmark_id": request_config.benchmark_id,
            "cancel_session_on_failure": request_config.cancel_session_on_failure,
            "sampling_params": request_config.sampling_params,
        }

        # Write to file with lock for thread safety
        with self.lock:
            self.file_handle.write(json.dumps(request_data) + "\n")
            self.file_handle.flush()  # Ensure immediate write

    def close(self) -> None:
        """Close the file handle."""
        if self.file_handle is not None:
            with self.lock:
                self.file_handle.close()
                self.file_handle = None
            logger.info(f"Closed dispatched requests file: {self.output_file}")


class DispatchWorker:
    """Worker that schedules and dispatches requests to worker queue."""

    def __init__(
        self,
        input_queue: Queue,
        ready_queue: Queue,
        service_metrics: ServiceMetrics,
        scheduler: DispatchScheduler,
        req_launcher: RequestsLauncher,
        benchmark_context: BenchmarkContext,
        worker_context: WorkerContext,
        request_writer: DispatchedRequestWriter,
    ):
        self.input_queue = input_queue
        self.ready_queue = ready_queue
        self.service_metrics = service_metrics
        self.scheduler = scheduler
        self.req_launcher = req_launcher
        self.benchmark_context = benchmark_context
        self.worker_context = worker_context
        self.dispatcher = RequestDispatcher(
            input_queue=input_queue,
            service_metrics=service_metrics,
            benchmark_id=benchmark_context.benchmark_id,
            telemetry_enabled=benchmark_context.telemetry_enabled,
            request_writer=request_writer,
        )

    def run(self) -> None:
        """Main worker loop."""
        logger.info(f"Dispatch worker {self.worker_context.worker_id} starting")

        while not self.worker_context.stop_event.is_set():
            # Check if there's a request ready to dispatch from scheduler
            ready = self.scheduler.pop_ready()
            if ready is not None:
                self.dispatcher.dispatch_request(ready)
                continue

            # Try to get a new request from the ready queue
            try:
                request_config = self.ready_queue.get(timeout=QUEUE_GET_TIMEOUT_S)
            except Empty:
                # No new requests, check if we should exit
                if self.worker_context.stop_event.is_set():
                    break
                continue

            # Add to scheduler (handles dispatch timing and session sequencing)
            self.scheduler.add_request(request_config)

            # Try to dispatch immediately if it's ready
            ready = self.scheduler.pop_ready()
            if ready is not None:
                self.dispatcher.dispatch_request(ready)

        # Drain any remaining ready requests from scheduler
        # Since multiple dispatcher threads may reach this point, they'll all try to drain
        # but that's okay - pop_ready is thread-safe and returns None when empty
        self._drain_scheduler()

        logger.info(f"Dispatch worker {self.worker_context.worker_id} exiting")

    def _drain_scheduler(self) -> None:
        """Drain any remaining ready requests from the scheduler."""
        logger.info(
            f"Dispatch worker {self.worker_context.worker_id}: draining scheduler before exit"
        )
        while True:
            ready = self.scheduler.pop_ready()
            if ready is None:
                break
            self.dispatcher.dispatch_request(ready)
