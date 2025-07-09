from abc import ABC, abstractmethod

from veeksha.config.generators.interval_generator.base_generator import (
    BaseRequestIntervalGeneratorConfig,
)


class BaseRequestIntervalGenerator(ABC):
    def __init__(self, config: BaseRequestIntervalGeneratorConfig):
        self.config = config

    @abstractmethod
    def get_next_inter_request_time(self) -> float:
        pass
