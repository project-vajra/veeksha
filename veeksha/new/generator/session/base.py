from abc import abstractmethod
from typing import Optional

from veeksha.new.config.generator.session import BaseSessionGeneratorConfig
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.session import Session


class BaseSessionGenerator:
    def __init__(
        self,
        config: BaseSessionGeneratorConfig,
        seed_manager: Optional[SeedManager] = None,
    ):
        self.config = config
        self.seed_manager = seed_manager or SeedManager(0)

    @abstractmethod
    def generate_session(self) -> Session:
        pass

    @abstractmethod
    def capacity(self) -> int:
        """Total number of sessions producible if finite; -1 if unbounded."""
