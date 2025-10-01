from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.interval_generator.base_generator import (
    BaseRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.interval_generator.poisson_generator import (
    PoissonRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.length_generator.base_generator import (
    BaseRequestLengthGeneratorConfig,
)
from veeksha.config.generators.length_generator.trace_generator import (
    TraceRequestLengthGeneratorConfig,
)
from veeksha.config.generators.request_generator.base_generator import (
    BaseRequestGeneratorConfig,
)
from veeksha.types.request_generator_type import RequestGeneratorType


@frozen_dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=TraceRequestLengthGeneratorConfig
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig
    )

    @classmethod
    def get_type(cls):
        return RequestGeneratorType.SYNTHETIC
