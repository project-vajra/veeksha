"""Prefetch worker for session generation and scheduling."""

import threading
from typing import Optional

from veeksha.logger import init_logger
from veeksha.new.core.context import WorkerContext
from veeksha.new.generator.session.base import BaseSessionGenerator
from veeksha.new.traffic.base import BaseTrafficScheduler

logger = init_logger(__name__)


class PrefetchWorker:
    """Worker that generates sessions and schedules them with the traffic scheduler.

    This worker pulls sessions from the session generator and feeds them to the
    traffic scheduler, which then manages the dispatch timing of individual requests.
    """

    # TODO this would probably throttle high frequency bursts of sessions, add an increasing function
    _POLL_INTERVAL_S = 0.05

    def __init__(
        self,
        traffic_scheduler: BaseTrafficScheduler,
        session_generator: BaseSessionGenerator,
        generator_lock: threading.Lock,
        worker_context: WorkerContext,
        max_sessions: int = -1,
    ):
        """Initialize the prefetch worker.

        Args:
            traffic_scheduler: Scheduler to schedule sessions with
            session_generator: Generator to get sessions from
            generator_lock: Lock protecting the session generator
            worker_context: Worker context with stop event
            max_sessions: Maximum sessions to generate (-1 for unlimited)
        """
        self.traffic_scheduler = traffic_scheduler
        self.session_generator = session_generator
        self.generator_lock = generator_lock
        self.worker_context = worker_context
        self.max_sessions = max_sessions
        self.sessions_generated = 0

    def _has_capacity(self) -> bool:
        """Check if we can generate more sessions."""
        if self.max_sessions < 0:
            # Check generator capacity
            capacity = self.session_generator.capacity()
            if capacity < 0:
                return True  # Unlimited
            return self.sessions_generated < capacity
        return self.sessions_generated < self.max_sessions

    def _generate_session(self) -> Optional[object]:
        """Generate next session in a thread-safe manner."""
        while not self.worker_context.stop_event.is_set():
            with self.generator_lock:
                if not self._has_capacity():
                    return None  # Exhausted

                try:
                    session = self.session_generator.generate_session()
                    self.sessions_generated += 1
                    return session
                except StopIteration:
                    logger.debug(
                        "Prefetch worker %s: generator exhausted",
                        self.worker_context.worker_id,
                    )
                    return None

        return None

    def run(self) -> None:
        """Main worker loop."""
        logger.debug("Prefetch worker %s starting", self.worker_context.worker_id)

        while not self.worker_context.stop_event.is_set():
            session = self._generate_session()
            if session is None:
                break

            # Schedule the session with traffic scheduler
            self.traffic_scheduler.schedule_session(session)

            if self.sessions_generated % 100 == 0:
                logger.debug(
                    "Prefetch progress: %d sessions generated",
                    self.sessions_generated,
                )

        logger.debug("Prefetch worker %s exiting", self.worker_context.worker_id)
