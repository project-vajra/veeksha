"""Prefetch worker for session generation and scheduling."""

import threading
from typing import Optional

from veeksha.logger import init_logger
from veeksha.new.core.context import WorkerContext
from veeksha.new.core.session import Session
from veeksha.new.generator.session.base import BaseSessionGenerator
from veeksha.new.traffic.base import BaseTrafficScheduler

logger = init_logger(__name__)


class SharedSessionCounter:
    """Thread-safe shared counter for tracking sessions across workers."""

    def __init__(self, max_sessions: int = -1):
        self.max_sessions = max_sessions
        self._count = 0

    def try_increment(self) -> bool:
        if self.max_sessions < 0:
            self._count += 1
            return True
        if self._count < self.max_sessions:
            self._count += 1
            return True
        return False

    @property
    def count(self) -> int:
        return self._count


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
        session_counter: SharedSessionCounter,
    ):
        """Initialize the prefetch worker.

        Args:
            traffic_scheduler: Scheduler to schedule sessions with
            session_generator: Generator to get sessions from
            generator_lock: Lock protecting the session generator
            worker_context: Worker context with stop event
            session_counter: Shared counter for tracking sessions across workers
        """
        self.traffic_scheduler = traffic_scheduler
        self.session_generator = session_generator
        self.generator_lock = generator_lock
        self.worker_context = worker_context
        self.session_counter = session_counter

    def _generate_session(self) -> Optional[Session]:
        """Generate next session in a thread-safe manner."""
        while not self.worker_context.stop_event.is_set():
            with self.generator_lock:
                if not self.session_counter.try_increment():
                    return None  # exhausted

                try:
                    session = self.session_generator.generate_session()
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
                logger.info(
                    "Prefetch worker %s: no more sessions to generate",
                    self.worker_context.worker_id,
                )
                break

            logger.info(
                "Generated session %s with %d requests",
                session.id,
                len(session.session_graph.nodes),
            )

            # Schedule the session with traffic scheduler
            self.traffic_scheduler.schedule_session(session)
            logger.info("Scheduled session %s", session.id)

            if self.session_counter.count % 100 == 0:
                logger.debug(
                    "Prefetch progress: %d sessions generated",
                    self.session_counter.count,
                )

        logger.debug("Prefetch worker %s exiting", self.worker_context.worker_id)
