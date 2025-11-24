from veeksha.config.generators.interval_generator.constant_generator import (
    ConstantRequestIntervalGeneratorConfig,
)
from veeksha.generators.interval_generator.base_generator import (
    BaseRequestIntervalGenerator,
)


class ConstantRequestIntervalGenerator(BaseRequestIntervalGenerator):
    def __init__(
        self,
        config: ConstantRequestIntervalGeneratorConfig,
        rng=None,
    ):
        self.config = config
        self.qps = self.config.qps
        self.interval = 1.0 / self.qps

    def get_next_inter_request_time(self) -> float:
        return self.interval
