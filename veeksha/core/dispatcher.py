"""Request dispatcher for sending requests to worker queues."""

import logging
import random
import time
from queue import Queue
from typing import List

from veeksha.core.context import WorkerContext
from veeksha.dashboard.events import RequestStartedEvent
from veeksha.dashboard.handler import emit_dashboard_event
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics

logger = init_logger(__name__)


class RequestDispatcher:
    """Handles dispatching of requests to worker queues using power-of-two load balancing."""

    def __init__(
        self,
        input_queues: List[Queue],
        service_metrics: ServiceMetrics,
        benchmark_id: str,
        telemetry_enabled: bool,
        worker_contexts: List[WorkerContext],
    ):
        self.input_queues = input_queues
        self.service_metrics = service_metrics
        self.benchmark_id = benchmark_id
        self.telemetry_enabled = telemetry_enabled
        self.worker_contexts = worker_contexts

    def _select_worker_power_of_two(self) -> WorkerContext:
        """Select worker using power-of-two load balancing.

        Randomly samples two workers and chooses the one with lower load.
        This provides good load distribution with minimal overhead.
        """
        n = len(self.worker_contexts)
        if n == 0:
            raise ValueError("No worker contexts available for load balancing")
        if n == 1:
            return self.worker_contexts[0]

        # Sample two random workers
        idx1 = random.randint(0, n - 1)
        idx2 = random.randint(0, n - 1)
        # Ensure we sample two different workers
        while idx2 == idx1 and n > 1:
            idx2 = random.randint(0, n - 1)

        worker1 = self.worker_contexts[idx1]
        worker2 = self.worker_contexts[idx2]

        # Choose the worker with lower load
        if worker1.get_load() <= worker2.get_load():
            return worker1
        else:
            return worker2

    def dispatch_request(self, request_config) -> None:
        """Dispatch a single request to workers using power-of-two load balancing."""
        self.service_metrics.register_launched_request()
        request_config.benchmark_id = self.benchmark_id

        # Power-of-two load balancing: select least loaded worker
        selected_worker = self._select_worker_power_of_two()
        if self.telemetry_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Dispatched request {request_config.id} to worker {selected_worker.worker_id} "
                f"(load: {selected_worker.get_load()})"
            )

        self.input_queues[selected_worker.worker_id].put(request_config)

        # Emit dashboard event
        if request_config.id is not None:
            emit_dashboard_event(
                RequestStartedEvent(
                    request_id=request_config.id,
                    timestamp=time.time(),
                    input_tokens=request_config.prompt[1],
                    benchmark_id=self.benchmark_id,
                )
            )
