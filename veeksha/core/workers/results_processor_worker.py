"""Worker that processes results from the output queue."""

import threading
import time
from queue import Queue
from typing import List, Tuple

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
    ):
        self.output_queue = output_queue
        self.service_metrics = service_metrics
        self.generated_responses = generated_responses
        self.responses_lock = responses_lock
        self.pbar = pbar
        self.pbar_lock = pbar_lock
        self.scheduler = scheduler
        self.worker_context = worker_context

    def run(self) -> None:
        """Main worker loop."""
        logger.debug("Results processor worker %s starting", self.worker_context.worker_id)

        while not self.worker_context.stop_event.is_set():
            result = self.output_queue.get()
            if result is None:
                break
            if self.service_metrics.num_completed_requests % 1000 == 0:
                logger.debug(
                    f"Results processor progress: {self.service_metrics.num_completed_requests} requests completed"
                )

            self.process_result(result)
            
        logger.debug("Results processor worker %s exiting", self.worker_context.worker_id)

    def process_result(self, result: Tuple[RequestMetrics, Response]) -> None:
        """Process a result from the output queue."""
        request_metrics, generated_response = result
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
