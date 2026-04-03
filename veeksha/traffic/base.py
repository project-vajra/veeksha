from __future__ import annotations

import threading
from abc import abstractmethod
from typing import TYPE_CHECKING, Mapping, Optional, Set, Tuple

from veeksha.config.traffic import BaseTrafficConfig
from veeksha.core.request import Request
from veeksha.core.response import ChannelResponse
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.types import ChannelModality

if TYPE_CHECKING:
    from veeksha.traffic.dispatch_tracker import DispatchTracker


class BaseTrafficScheduler:
    def __init__(self, config: BaseTrafficConfig, seed_manager: SeedManager):
        self.config = config
        self.seed_manager = seed_manager
        self._decoding_ids: Set[int] = set()
        self._decoding_lock = threading.Lock()

    @abstractmethod
    def schedule_session(self, session: Session) -> None:
        """Schedule a session for dispatch."""
        raise NotImplementedError

    @abstractmethod
    def pop_ready(self) -> Optional[Tuple[Request, int, int]]:
        """Pop a ready request from the scheduler.

        Returns:
            Tuple of (request, session_id, session_size) if a request is ready,
            None otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def wait_for_ready(
        self, timeout: float = 0.001
    ) -> Optional[Tuple[Request, int, int]]:
        """Wait for a ready request with timeout.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Tuple of (request, session_id, session_size) if ready, None if timeout.
        """
        raise NotImplementedError

    @abstractmethod
    def notify_completion(
        self,
        request_id: int,
        completed_at_monotonic: float,
        success: bool,
        channel_responses: Optional[Mapping[ChannelModality, ChannelResponse]] = None,
    ) -> None:
        """Notify the scheduler that a request has completed."""
        raise NotImplementedError

    @abstractmethod
    def get_session_id(self, request_id: int) -> int:
        """Get the session ID for a given request ID.

        Returns -1 if the request is not found.
        """
        raise NotImplementedError

    @abstractmethod
    def get_session_size(self, request_id: int) -> int:
        """Get the total number of requests in the session for a given request ID.

        Returns 1 if the request is not found.
        """
        raise NotImplementedError

    @abstractmethod
    def has_pending_work(self) -> bool:
        """Check if there are pending sessions or in-flight requests."""
        raise NotImplementedError

    @abstractmethod
    def get_in_flight_request_ids(self) -> Set[int]:
        """Return the set of request IDs currently in-flight."""
        raise NotImplementedError

    @property
    def dispatch_tracker(self) -> Optional[DispatchTracker]:
        """Optional dispatch tracker for ticket-based ordering."""
        return None

    def notify_request_sent(self, request_id: int) -> None:
        """Called when the first content chunk is received (prefill complete).

        Tracks the request as actively decoding. Subclasses (e.g.
        :class:`SequentialLaunchTrafficScheduler`) may add further logic.
        """
        with self._decoding_lock:
            self._decoding_ids.add(request_id)

    def notify_request_done_decoding(self, request_id: int) -> None:
        """Remove a request from the actively-decoding set."""
        with self._decoding_lock:
            self._decoding_ids.discard(request_id)

    def get_decoding_count(self) -> int:
        """Return the number of requests currently in the decode phase."""
        with self._decoding_lock:
            return len(self._decoding_ids)

    def reset_reference_time(self) -> None:
        """Optional hook invoked before the benchmark starts dispatching."""
        return
