from abc import abstractmethod
from typing import Mapping, Optional, Set, Tuple

from veeksha.config.traffic import BaseTrafficConfig
from veeksha.core.request import Request
from veeksha.core.response import ChannelResponse
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.types import ChannelModality


class BaseTrafficScheduler:
    def __init__(self, config: BaseTrafficConfig, seed_manager: SeedManager):
        self.config = config
        self.seed_manager = seed_manager

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

    def reset_reference_time(self) -> None:
        """Optional hook invoked before the benchmark starts dispatching."""
        return
