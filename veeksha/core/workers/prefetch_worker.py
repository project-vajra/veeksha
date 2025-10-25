"""Worker that generates requests and pushes them to the ready queue."""

import threading
from queue import Queue

from veeksha.core.context import WorkerContext
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics

logger = init_logger(__name__)


class PrefetchWorker:
    """Worker that generates requests and pushes them to the ready queue."""

    def __init__(
        self,
        ready_queue: Queue,
        service_metrics: ServiceMetrics,
        request_generator: BaseRequestGenerator,
        generator_lock: threading.Lock,
        worker_context: WorkerContext,
    ):
        self.ready_queue = ready_queue
        self.service_metrics = service_metrics
        self.request_generator = request_generator
        self.generator_lock = generator_lock
        self.worker_context = worker_context

    def run(self) -> None:
        """Main worker loop."""
        logger.info(f"Prefetch worker {self.worker_context.worker_id} starting")

        while not self.worker_context.stop_event.is_set():
            request_config = self._generate_request()
            if request_config is None:
                break

            # Push to ready queue (outside lock to avoid holding it too long)
            self.ready_queue.put(request_config)

            if self.service_metrics.num_generated_requests % 1000 == 0:
                logger.debug(
                    f"Prefetch progress: {self.service_metrics.num_generated_requests} requests generated"
                )

        logger.info(f"Prefetch worker {self.worker_context.worker_id} exiting")

    def _generate_request(self):
        """Generate next request in a thread-safe manner. Returns None if should stop."""
        with self.generator_lock:
            # Check if we've generated enough requests
            if (
                self.service_metrics.num_generated_requests
                >= self.service_metrics.max_requests
            ):
                logger.info(
                    f"Prefetch worker {self.worker_context.worker_id}: max requests reached"
                )
                return None

            try:
                request_config = self.request_generator.get_request()
            except StopIteration:
                logger.info(
                    f"Prefetch worker {self.worker_context.worker_id}: generator exhausted"
                )
                return None

            # Handle special sentinel values
            if request_config.dispatch_delay == -1:
                logger.info(
                    f"Prefetch worker {self.worker_context.worker_id}: stop policy triggered"
                )
                self.service_metrics.request_stop()
                self.worker_context.stop_event.set()
                return None
            elif request_config.dispatch_delay < 0:
                raise ValueError(
                    f"Invalid dispatch_delay '{request_config.dispatch_delay}' from generator"
                )

            self.service_metrics.num_generated_requests += 1
            return request_config
