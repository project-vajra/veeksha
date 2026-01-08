from veeksha.config.generator.interval import (
    FixedIntervalGeneratorConfig,
)
from veeksha.generator.interval.base import BaseIntervalGenerator


class FixedIntervalGenerator(BaseIntervalGenerator):
    def __init__(self, config: FixedIntervalGeneratorConfig, rng: None):
        self.config = config

    def get_next_interval(self) -> float:
        return self.config.interval
