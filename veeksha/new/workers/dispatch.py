"""Dispatch worker for sending requests to client queues."""

import random
import time
from queue import Queue
from typing import List

from veeksha.logger import init_logger
from veeksha.new.core.context import WorkerContext
from veeksha.new.evaluator.base import BaseEvaluator
from veeksha.new.traffic.base import BaseTrafficScheduler

logger = init_logger(__name__)


class DispatchWorker:
    """Worker that polls ready requests from scheduler and dispatches to client queues.

    This worker:
    1. Polls the traffic scheduler for ready requests (pop_ready)
    2. Registers requests with the evaluator
    3. Dispatches requests to client worker queues
    """

    _POLL_INTERVAL_S = 0.001  # Fast polling for responsiveness

    def __init__(
        self,
        traffic_scheduler: BaseTrafficScheduler,
        client_queues: List[Queue],
        evaluator: BaseEvaluator,
        worker_context: WorkerContext,
    ):
        """Initialize the dispatch worker.

        Args:
            traffic_scheduler: Scheduler to poll for ready requests
            client_queues: Queues to dispatch requests to (one per client worker)
            evaluator: Evaluator for registering request dispatch
            worker_context: Worker context with stop event
        """
        self.traffic_scheduler = traffic_scheduler
        self.client_queues = client_queues
        self.evaluator = evaluator
        self.worker_context = worker_context

    def _select_queue(self) -> Queue:
        """Select a client queue using power-of-two load balancing"""
        n = len(self.client_queues)
        if n == 1:
            return self.client_queues[0]

        idx1, idx2 = random.sample(range(n), 2)

        q1 = self.client_queues[idx1]
        q2 = self.client_queues[idx2]

        return q1 if q1.qsize() <= q2.qsize() else q2

    def run(self) -> None:
        """Main worker loop."""
        logger.debug("Dispatch worker %s starting", self.worker_context.worker_id)

        while not self.worker_context.stop_event.is_set():
            # Poll for ready request
            request = self.traffic_scheduler.pop_ready()

            if request is None:
                # No request ready, small sleep
                time.sleep(self._POLL_INTERVAL_S)
                continue

            dispatched_at = time.monotonic()

            # Get session info from scheduler's internal tracking
            session_id = self._get_session_id(request)
            session_size = self._get_session_size(request)

            # Register with evaluator
            self.evaluator.register_request(
                request_id=request.id,
                session_id=session_id,
                dispatched_at=dispatched_at,
                channels=request.channels,
            )

            # Dispatch to client queue (request, session_id, session_size, dispatched_at)
            queue = self._select_queue()
            queue.put((request, session_id, session_size, dispatched_at))

        # Drain remaining ready requests
        self._drain()

        logger.debug("Dispatch worker %s exiting", self.worker_context.worker_id)

    def _get_session_id(self, request) -> int:
        """Get session ID for a request."""
        return self.traffic_scheduler.get_session_id(request.id)

    def _get_session_size(self, request) -> int:
        """Get total number of requests in the session."""
        return self.traffic_scheduler.get_session_size(request.id)

    def _drain(self) -> None:
        """Drain any remaining ready requests."""
        logger.debug(
            "Dispatch worker %s: draining scheduler", self.worker_context.worker_id
        )
        while True:
            request = self.traffic_scheduler.pop_ready()
            if request is None:
                break

            dispatched_at = time.monotonic()
            session_id = self._get_session_id(request)
            session_size = self._get_session_size(request)

            self.evaluator.register_request(
                request_id=request.id,
                session_id=session_id,
                dispatched_at=dispatched_at,
                channels=request.channels,
            )

            queue = self._select_queue()
            queue.put((request, session_id, session_size, dispatched_at))
