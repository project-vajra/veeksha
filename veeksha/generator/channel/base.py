from abc import abstractmethod
from typing import Any

from veeksha.config.generator.channel import BaseChannelGeneratorConfig
from veeksha.core.seeding import SeedManager


class BaseChannelGenerator:
    def __init__(self, config: BaseChannelGeneratorConfig, seed_manager: SeedManager):
        self.config = config
        self.seed_manager = seed_manager

    @abstractmethod
    def generate_content(self, is_root: bool = False) -> Any:
        pass
