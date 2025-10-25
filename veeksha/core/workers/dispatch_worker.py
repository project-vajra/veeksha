"""Worker that schedules and dispatches requests to worker queue."""

from queue import Empty, Queue

from veeksha.core.context import BenchmarkContext, WorkerContext
from veeksha.core.dispatch_scheduler import DispatchScheduler
from veeksha.core.dispatcher import RequestDispatcher
from veeksha.core.requests_launcher import RequestsLauncher
from veeksha.core.workers.constants import QUEUE_GET_TIMEOUT_S
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics

logger = init_logger(__name__)


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
