import math

import numpy as np

from veeksha.config.generators.interval_generator.poisson_generator import (
    PoissonRequestIntervalGeneratorConfig,
)
from veeksha.generators.interval_generator.base_generator import (
    BaseRequestIntervalGenerator,
)


class PoissonRequestIntervalGenerator(BaseRequestIntervalGenerator):
    def __init__(
        self,
        config: PoissonRequestIntervalGeneratorConfig,
        rng: np.random.RandomState,
    ):
        self.config = config
        self.rng = rng

        self.qps = self.config.qps
        self.std = 1.0 / self.qps
        self.max_interval = self.std * 3.0

    def get_next_inter_request_time(self) -> float:
        next_interval = -math.log(1.0 - float(self.rng.random_sample())) / self.qps
        next_interval = min(next_interval, self.max_interval)
        return next_interval
