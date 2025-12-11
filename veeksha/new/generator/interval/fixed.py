from veeksha.new.config.generator.interval import (
    FixedIntervalGeneratorConfig,
)
from veeksha.new.generator.interval.base import BaseIntervalGenerator


class FixedIntervalGenerator(BaseIntervalGenerator):
    def __init__(self, config: FixedIntervalGeneratorConfig, rng: None):
        self.config = config

    def get_next_interval(self) -> float:
        return self.config.interval
