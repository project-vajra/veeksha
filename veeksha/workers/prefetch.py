"""Prefetch worker for session generation and scheduling."""

import threading
import time
from typing import List, Optional

from veeksha.core.context import WorkerContext
from veeksha.core.session import Session
from veeksha.core.workload_fingerprint import WorkloadFingerprint
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
        pregenerated_sessions: Optional[List[Session]] = None,
        workload_fingerprint: Optional[WorkloadFingerprint] = None,
        fingerprint_pregenerated: bool = False,
    ):
        """Initialize the prefetch worker.

        Args:
            traffic_scheduler: Scheduler to schedule sessions with
            session_generator: Generator to get sessions from
            generator_lock: Lock protecting the session generator
            worker_context: Worker context with stop event
            session_counter: Shared counter for tracking sessions across workers
            pregenerated_sessions: Optional list of pre-generated sessions to use
            workload_fingerprint: Optional hasher fed in generation order under
                ``generator_lock``. Not thread-safe on its own; the lock is the
                only serialization it needs.
            fingerprint_pregenerated: When True, pregenerated sessions were
                already hashed during pregeneration — do not double-count them.
        """
        self.traffic_scheduler = traffic_scheduler
        self.session_generator = session_generator
        self.generator_lock = generator_lock
        self.worker_context = worker_context
        self.session_counter = session_counter
        self._pregenerated_sessions = pregenerated_sessions
        self._pregenerated_index = 0
        self._workload_fingerprint = workload_fingerprint
        self._fingerprint_pregenerated = fingerprint_pregenerated

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

    def _record_session(self, session: Session) -> None:
        """Feed one session into the workload hasher. Caller holds the lock."""
        if self._workload_fingerprint is not None:
            self._workload_fingerprint.add_session(session)

    def _generate_session(self) -> Optional[Session]:
        """Generate next session in a thread-safe manner."""
        # If we have pre-generated sessions, use those
        if self._pregenerated_sessions is not None:
            with self.generator_lock:
                if self._pregenerated_index >= len(self._pregenerated_sessions):
                    return None
                session = self._pregenerated_sessions[self._pregenerated_index]
                self._pregenerated_index += 1
                self.session_counter._count += 1
                # Pre-gen path already fingerprinted during pregeneration when
                # fingerprint_pregenerated is set; otherwise hash on consume so
                # the sequence still matches generation order.
                if not self._fingerprint_pregenerated:
                    self._record_session(session)
                return session

        # Otherwise generate on-the-fly
        while not self.worker_context.stop_event.is_set():
            with self.generator_lock:
                if not self.session_counter.try_increment():
                    return None  # exhausted

                try:
                    session = self.session_generator.generate_session()
                    self._record_session(session)
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
