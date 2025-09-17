from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.interval_generator.base_generator_config import (
    BaseRequestIntervalGeneratorConfig,
)
from veeksha.types.request_interval_generator_type import RequestIntervalGeneratorType


@frozen_dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @classmethod
    def get_type(cls):
        return RequestIntervalGeneratorType.STATIC
