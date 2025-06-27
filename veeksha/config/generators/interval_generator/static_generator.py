from dataclasses import field
from veeksha.types.request_interval_generator_type import RequestIntervalGeneratorType
from veeksha.config.generators.interval_generator.base_generator import BaseRequestIntervalGeneratorConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @classmethod
    def get_type(cls):
        return RequestIntervalGeneratorType.STATIC