import numpy as np
from scipy.stats import gamma

from veeksha.new.config.generator.interval import (
    GammaIntervalGeneratorConfig,
)
from veeksha.new.generator.interval.base import BaseIntervalGenerator


class GammaIntervalGenerator(BaseIntervalGenerator):
    def __init__(
        self,
        config: GammaIntervalGeneratorConfig,
        rng: np.random.RandomState,
    ):
        self.config = config
        self.rng = rng

        cv = self.config.cv
        self.arrival_rate = self.config.arrival_rate
        self.gamma_shape = 1.0 / (cv**2)

    def get_next_interval(self) -> float:
        gamma_scale = 1.0 / (self.arrival_rate * self.gamma_shape)
        return gamma.rvs(self.gamma_shape, scale=gamma_scale, random_state=self.rng)
