from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.interval_generator.base_generator_config import (
    BaseRequestIntervalGeneratorConfig,
)
from veeksha.types.request_interval_generator_type import RequestIntervalGeneratorType


@frozen_dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=1.0,
        metadata={"help": "Queries per second for the Poisson distribution."},
    )

    @classmethod
    def get_type(cls):
        return RequestIntervalGeneratorType.POISSON
