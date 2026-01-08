from abc import ABC, abstractmethod

from veeksha.config.generator.interval import (
    BaseIntervalGeneratorConfig,
)


class BaseIntervalGenerator(ABC):
    def __init__(self, config: BaseIntervalGeneratorConfig, rng=None):
        """Base class for interval generators.

        Args:
            config: Configuration dataclass.
            rng: Optional random generator to use for sampling.
        """
        self.config = config
        self.rng = rng

    @abstractmethod
    def get_next_interval(self) -> float:
        pass
