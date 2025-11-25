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

    _GENERATION_BUDGET_POLL_INTERVAL_S = 0.05

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

    def _has_generation_budget(self) -> bool:
        """Return True if we can generate more requests right now."""
        return self.service_metrics.num_generated_requests < (
            self.service_metrics.max_requests
            + self.service_metrics.num_errored_requests
            + self.service_metrics.num_cancelled_requests
        )

    def run(self) -> None:
        """Main worker loop."""
        logger.debug("Prefetch worker %s starting", self.worker_context.worker_id)

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

        logger.debug("Prefetch worker %s exiting", self.worker_context.worker_id)

    def _generate_request(self):
        """Generate next request in a thread-safe manner. Returns None if should stop."""
        while not self.worker_context.stop_event.is_set():
            wait_for_budget = False
            with self.generator_lock:
                if not self._has_generation_budget():
                    wait_for_budget = True
                else:
                    try:
                        request_config = self.request_generator.get_request()
                    # error exhaustion policy is active
                    except StopIteration:
                        self.service_metrics.notify_error(
                            RuntimeError(
                                "Request generator exhausted with 'error' policy. Aborting benchmark."
                            )
                        )
                        self.worker_context.stop_event.set()
                        return None

                    # stop exhaustion policy is active
                    stop_signal = request_config.wait_after_prev_response_s
                    if stop_signal == -1:
                        logger.debug(
                            "Prefetch worker %s: stop policy triggered",
                            self.worker_context.worker_id,
                        )
                        self.service_metrics.request_stop()
                        self.worker_context.stop_event.set()
                        return None
                    elif stop_signal is not None and stop_signal < 0:
                        raise ValueError(
                            f"Invalid wait_after_prev_response_s '{stop_signal}' from generator"
                        )

                    self.service_metrics.num_generated_requests += 1
                    return request_config

            if wait_for_budget:
                triggered = self.worker_context.stop_event.wait(
                    self._GENERATION_BUDGET_POLL_INTERVAL_S
                )
                if triggered:
                    break
                continue

        return None
