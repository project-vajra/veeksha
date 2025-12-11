from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.new.config.generator.interval import (
    BaseIntervalGeneratorConfig,
    PoissonIntervalGeneratorConfig,
)
from veeksha.new.config.generator.length import (
    BaseLengthGeneratorConfig,
    UniformLengthGeneratorConfig,
)
from veeksha.new.types import SessionGraphType


@frozen_dataclass
class BaseSessionGraphGeneratorConfig(BasePolyConfig):
    pass


@frozen_dataclass
class LinearSessionGraphGeneratorConfig(BaseSessionGraphGeneratorConfig):
    """
    Generator of linear request graphs (a sequence of requests).
    """

    # TODO: add generator for the number of requests
    num_request_generator: BaseLengthGeneratorConfig = field(
        default_factory=UniformLengthGeneratorConfig,
        metadata={
            "help": "The generator for the number of requests. Available: uniform"
        },
    )
    request_wait_generator: BaseIntervalGeneratorConfig = field(
        default_factory=PoissonIntervalGeneratorConfig,
        metadata={"help": "The generator for the wait time between requests."},
    )

    @classmethod
    def get_type(cls):
        return SessionGraphType.LINEAR
