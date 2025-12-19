"""Prefetch worker for session generation and scheduling."""

import math
import threading
import time
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

    # starts fast and decays to max over time
    _MIN_POLL_INTERVAL_S = 0.001
    _MAX_POLL_INTERVAL_S = 0.05
    _DECAY_RATE = 0.1  # higher = faster decay

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
        self._sessions_generated = 0

    def _get_poll_interval(self) -> float:
        """Calculate poll interval with exponential decay.

        Starts at _MIN_POLL_INTERVAL_S and exponentially decays toward
        _MAX_POLL_INTERVAL_S as more sessions are generated. This allows
        fast bursting at startup while settling into a steady-state rate.

        Returns:
            Poll interval in seconds.
        """
        decay = math.exp(-self._DECAY_RATE * self._sessions_generated)
        interval = (
            self._MAX_POLL_INTERVAL_S
            - (self._MAX_POLL_INTERVAL_S - self._MIN_POLL_INTERVAL_S) * decay
        )
        return interval

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
            self._sessions_generated += 1
            logger.info("Scheduled session %s", session.id)

            if self.session_counter.count % 100 == 0:
                logger.debug(
                    "Prefetch progress: %d sessions generated",
                    self.session_counter.count,
                )

            # Throttle with decaying poll interval (fast start, slow steady-state)
            time.sleep(self._get_poll_interval())

        logger.debug("Prefetch worker %s exiting", self.worker_context.worker_id)
