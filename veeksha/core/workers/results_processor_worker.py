"""Worker that processes results from the output queue."""

import json
import os
import threading
import time
from queue import Queue
from typing import List, Optional, Tuple

from tqdm import tqdm

from veeksha.core.context import WorkerContext
from veeksha.core.dispatch_scheduler import DispatchScheduler
from veeksha.core.response import Response
from veeksha.dashboard.events import RequestCompletedEvent
from veeksha.dashboard.handler import emit_dashboard_event
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics
from veeksha.metrics.service_metrics import ServiceMetrics

logger = init_logger(__name__)


class RequestMetricsWriter:
    """Writes request metrics to a JSONL file in streaming fashion."""

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
            logger.info(f"Request metrics will be written to: {output_file}")

    def write_metrics(self, request_metrics: RequestMetrics) -> None:
        """Write request metrics to the file.

        Args:
            request_metrics: The RequestMetrics object to write
        """
        if not self.enabled or self.file_handle is None:
            return

        # Extract all metrics data
        metrics_data = {
            "request_id": request_metrics.request_id,
            "session_id": request_metrics.session_id,
            "benchmark_id": request_metrics.benchmark_id,
            "request_dispatched_at": request_metrics.request_dispatched_at,
            "num_prompt_tokens": request_metrics.num_prompt_tokens,
            "num_output_tokens": request_metrics.num_output_tokens,
            "num_total_tokens": request_metrics.num_total_tokens,
            "ttft": request_metrics.ttft,
            "tpot": request_metrics.tpot,
            "end_to_end_latency": request_metrics.end_to_end_latency,
            "normalized_end_to_end_latency": request_metrics.normalized_end_to_end_latency,
            "output_throughput": request_metrics.output_throughput,
            "inter_token_times": request_metrics.inter_token_times,
            "error_msg": request_metrics.error_msg,
            "error_code": request_metrics.error_code,
        }

        # Write to file with lock for thread safety
        with self.lock:
            self.file_handle.write(json.dumps(metrics_data) + "\n")
            self.file_handle.flush()  # Ensure immediate write

    def close(self) -> None:
        """Close the file handle."""
        if self.file_handle is not None:
            with self.lock:
                self.file_handle.close()
                self.file_handle = None
            logger.info(f"Closed request metrics file: {self.output_file}")


class ResultsProcessorWorker:
    """Worker that processes results from the output queue."""

    def __init__(
        self,
        output_queue: Queue,
        service_metrics: ServiceMetrics,
        generated_responses: List[Response],
        responses_lock: threading.Lock,
        pbar: tqdm,
        pbar_lock: threading.Lock,
        scheduler: DispatchScheduler,
        worker_context: WorkerContext,
        metrics_writer: Optional[RequestMetricsWriter] = None,
    ):
        self.output_queue = output_queue
        self.service_metrics = service_metrics
        self.generated_responses = generated_responses
        self.responses_lock = responses_lock
        self.pbar = pbar
        self.pbar_lock = pbar_lock
        self.scheduler = scheduler
        self.worker_context = worker_context
        self.metrics_writer = metrics_writer

    def run(self) -> None:
        """Main worker loop."""
        logger.info(f"Results processor worker {self.worker_context.worker_id} starting")

        while not self.worker_context.stop_event.is_set():
            result = self.output_queue.get()
            if result is None:
                break
            if self.service_metrics.num_completed_requests % 1000 == 0:
                logger.debug(
                    f"Results processor progress: {self.service_metrics.num_completed_requests} requests completed"
                )

            self.process_result(result)
            
        logger.info(f"Results processor worker {self.worker_context.worker_id} exiting")

    def process_result(self, result: Tuple[RequestMetrics, Response]) -> None:
        """Process a result from the output queue."""
        request_metrics, generated_response = result

        # Write metrics to JSONL file if enabled
        if self.metrics_writer is not None:
            self.metrics_writer.write_metrics(request_metrics)

        self.service_metrics.add_request_metrics(request_metrics)
        # notify scheduler about completion for session-aware sequencing
        success = (
            getattr(request_metrics, "error_code", None) is None
            and getattr(request_metrics, "error_msg", None) is None
        )
        self.scheduler.notify_completion(
            request_id=request_metrics.request_id,
            completed_at_monotonic=time.monotonic(),
            success=success,
        )
        if generated_response is not None:
            with self.responses_lock:
                self.generated_responses.append(generated_response)

        # Emit completion event - ensure request_id is set
        assert (
            request_metrics.request_id is not None
        ), f"Request metrics has no ID: {request_metrics}"
        emit_dashboard_event(
            RequestCompletedEvent(
                request_id=str(request_metrics.request_id),
                timestamp=time.time(),
                final_metrics=request_metrics,
                benchmark_id=request_metrics.benchmark_id,
            )
        )

        with self.pbar_lock:
            self.pbar.update(self.service_metrics.num_completed_requests - self.pbar.n)
