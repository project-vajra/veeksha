from abc import abstractmethod
from typing import Any

from veeksha.new.config.generator.channel import BaseChannelGeneratorConfig
from veeksha.new.core.seeding import SeedManager


class BaseChannelGenerator:
    def __init__(self, config: BaseChannelGeneratorConfig, seed_manager: SeedManager):
        self.config = config
        self.seed_manager = seed_manager

    @abstractmethod
    def generate_content(self) -> Any:
        pass
