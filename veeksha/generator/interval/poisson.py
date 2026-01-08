import math

import numpy as np

from veeksha.config.generator.interval import (
    PoissonIntervalGeneratorConfig,
)
from veeksha.generator.interval.base import BaseIntervalGenerator


class PoissonIntervalGenerator(BaseIntervalGenerator):
    def __init__(
        self,
        config: PoissonIntervalGeneratorConfig,
        rng: np.random.RandomState,
    ):
        self.config = config
        self.rng = rng

        self.arrival_rate = self.config.arrival_rate
        self.std = 1.0 / self.arrival_rate
        self.max_interval = self.std * 3.0

    def get_next_interval(self) -> float:
        next_interval = (
            -math.log(1.0 - float(self.rng.random_sample())) / self.arrival_rate
        )
        next_interval = min(next_interval, self.max_interval)
        return next_interval
