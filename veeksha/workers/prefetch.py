"""Prefetch worker for session generation and scheduling."""

import threading
import time
from typing import Optional

from veeksha.core.context import WorkerContext
from veeksha.core.session import Session
from veeksha.generator.session.base import BaseSessionGenerator
from veeksha.logger import init_logger
from veeksha.traffic.base import BaseTrafficScheduler

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

    # unthrottled for first 3 seconds, then throttles
    _BURST_DURATION_S = 5.0
    _MAX_POLL_INTERVAL_S = 0.05

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

    def _get_poll_interval(self) -> float:
        """Calculate poll interval based on runtime duration.

        Unthrottled for the first _BURST_DURATION_S seconds, then throttles
        to _MAX_POLL_INTERVAL_S.

        Returns:
            Poll interval in seconds.
        """
        if time.monotonic() - self._start_time < self._BURST_DURATION_S:
            return 0.0
        return self._MAX_POLL_INTERVAL_S

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

        self._start_time = time.monotonic()

        while not self.worker_context.stop_event.is_set():
            session = self._generate_session()
            if session is None:
                logger.info(
                    "Prefetch worker %s: no more sessions to generate",
                    self.worker_context.worker_id,
                )
                break

            # Schedule the session with traffic scheduler
            self.traffic_scheduler.schedule_session(session)

            if self.session_counter.count % 100 == 0:
                logger.debug(
                    "Prefetch progress: %d sessions generated",
                    self.session_counter.count,
                )

            # Throttle (burst at start, then steady-state)
            time.sleep(self._get_poll_interval())

        logger.debug("Prefetch worker %s exiting", self.worker_context.worker_id)
