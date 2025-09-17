from abc import ABC, abstractmethod
from typing import Optional, Tuple

from veeksha.config.generators.length_generator.base_generator_config import (
    BaseRequestLengthGeneratorConfig,
)


class BaseRequestLengthGenerator(ABC):
    def __init__(self, config: BaseRequestLengthGeneratorConfig):
        self.config = config

    @abstractmethod
    def get_next_num_tokens(self) -> Tuple[float, float]:
        pass

    def capacity(self) -> Optional[int]:
        """Optional: total number of requests producible if finite; None if unbounded."""
        return None
