"""Request dispatcher for sending requests to worker queues."""

import logging
import time
from queue import Queue

from veeksha.dashboard.events import RequestStartedEvent
from veeksha.dashboard.handler import emit_dashboard_event
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics

logger = init_logger(__name__)


class RequestDispatcher:
    """Handles dispatching of requests to worker queues."""

    def __init__(
        self,
        input_queue: Queue,
        service_metrics: ServiceMetrics,
        benchmark_id: str,
        telemetry_enabled: bool,
    ):
        self.input_queue = input_queue
        self.service_metrics = service_metrics
        self.benchmark_id = benchmark_id
        self.telemetry_enabled = telemetry_enabled

    def dispatch_request(self, request_config) -> None:
        """Dispatch a single request to workers."""
        self.service_metrics.register_launched_request()
        request_config.benchmark_id = self.benchmark_id
        self.input_queue.put(request_config)

        if self.telemetry_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Dispatched request {request_config.id}")

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
