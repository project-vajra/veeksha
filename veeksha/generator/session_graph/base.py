from abc import abstractmethod
from typing import Optional

from veeksha.config.generator.session_graph import BaseSessionGraphGeneratorConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.session_graph import SessionGraph


class BaseSessionGraphGenerator:
    def __init__(
        self,
        config: BaseSessionGraphGeneratorConfig,
        seed_manager: Optional[SeedManager] = None,
    ):
        self.config = config
        self.seed_manager = seed_manager or SeedManager(0)

    @abstractmethod
    def generate_session_graph(self) -> SessionGraph:
        pass
