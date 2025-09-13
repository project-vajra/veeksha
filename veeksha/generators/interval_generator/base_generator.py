from abc import ABC, abstractmethod
from typing import Optional

from veeksha.config.generators.interval_generator.base_generator import (
    BaseRequestIntervalGeneratorConfig,
)


class BaseRequestIntervalGenerator(ABC):
    def __init__(self, config: BaseRequestIntervalGeneratorConfig):
        self.config = config

    @abstractmethod
    def get_next_inter_request_time(self) -> float:
        pass

    def capacity(self) -> Optional[int]:
        """Optional: total number of requests producible if finite; None if unbounded."""
        return None
