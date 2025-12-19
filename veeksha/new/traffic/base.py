from abc import abstractmethod
from typing import Optional

from veeksha.new.config.traffic import BaseTrafficConfig
from veeksha.new.core.request import Request
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.session import Session


class BaseTrafficScheduler:
    def __init__(self, config: BaseTrafficConfig, seed_manager: SeedManager):
        self.config = config
        self.seed_manager = seed_manager

    @abstractmethod
    def schedule_session(self, session: Session) -> None:
        """Schedule a session for dispatch."""
        raise NotImplementedError

    @abstractmethod
    def pop_ready(self) -> Optional[Request]:
        """Pop a ready request from the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def notify_completion(
        self, request_id: int, completed_at_monotonic: float, success: bool
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
