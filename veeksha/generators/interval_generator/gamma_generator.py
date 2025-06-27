from scipy.stats import gamma

from veeksha.config.config import GammaRequestIntervalGeneratorConfig
from veeksha.generators.interval_generator.base_generator import (
    BaseRequestIntervalGenerator,
)


class GammaRequestIntervalGenerator(BaseRequestIntervalGenerator):
    def __init__(self, config: GammaRequestIntervalGeneratorConfig):
        self.config = config

        cv = self.config.cv
        self.qps = self.config.qps
        self.gamma_shape = 1.0 / (cv**2)

    def get_next_inter_request_time(self) -> float:
        gamma_scale = 1.0 / (self.qps * self.gamma_shape)
        return gamma.rvs(self.gamma_shape, scale=gamma_scale)
