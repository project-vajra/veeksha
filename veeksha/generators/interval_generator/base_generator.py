from abc import ABC, abstractmethod
from typing import Optional

from veeksha.config.generators.interval_generator.base_generator import (
    BaseRequestIntervalGeneratorConfig,
)


class BaseRequestIntervalGenerator(ABC):
    def __init__(self, config: BaseRequestIntervalGeneratorConfig, rng=None):
        """Base class for interval generators.

        Args:
            config: Configuration dataclass.
            rng: Optional random generator to use for sampling.
        """
        self.config = config
        self.rng = rng

    @abstractmethod
    def get_next_inter_request_time(self) -> float:
        pass

    def capacity(self) -> Optional[int]:
        """Optional: total number of requests producible if finite; None if unbounded."""
        return None
