from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generator.interval import (
    BaseIntervalGeneratorConfig,
    PoissonIntervalGeneratorConfig,
)
from veeksha.config.generator.length import (
    BaseLengthGeneratorConfig,
    UniformLengthGeneratorConfig,
)
from veeksha.types import SessionGraphType


@frozen_dataclass
class BaseSessionGraphGeneratorConfig(BasePolyConfig):
    inherit_history: bool = field(
        default=True,
        metadata={
            "help": "Whether subsequent requests can inherit history from previous ones."
        },
    )


@frozen_dataclass
class LinearSessionGraphGeneratorConfig(BaseSessionGraphGeneratorConfig):
    """
    Generator of linear request graphs (a sequence of requests).
    """

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
