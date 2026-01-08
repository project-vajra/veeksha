from abc import ABC, abstractmethod

from veeksha.config.generator.length import (
    BaseLengthGeneratorConfig,
)


class BaseLengthGenerator(ABC):
    def __init__(self, config: BaseLengthGeneratorConfig, rng=None):
        self.config = config
        self.rng = rng

    @abstractmethod
    def get_next_value(self) -> int:
        pass
