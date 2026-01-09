from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.types import IntervalGeneratorType


@frozen_dataclass
class BaseIntervalGeneratorConfig(BasePolyConfig):
    pass


@frozen_dataclass
class GammaIntervalGeneratorConfig(BaseIntervalGeneratorConfig):
    arrival_rate: float = field(
        default=1.0, metadata={"help": "Arrival rate for the Gamma distribution."}
    )
    cv: float = field(
        default=0.5,
        metadata={"help": "Coefficient of variation for the Gamma distribution."},
    )

    @classmethod
    def get_type(cls):
        return IntervalGeneratorType.GAMMA


@frozen_dataclass
class PoissonIntervalGeneratorConfig(BaseIntervalGeneratorConfig):
    arrival_rate: float = field(
        default=1.0,
        metadata={"help": "Arrival rate for the Poisson distribution."},
    )

    @classmethod
    def get_type(cls):
        return IntervalGeneratorType.POISSON


@frozen_dataclass
class FixedIntervalGeneratorConfig(BaseIntervalGeneratorConfig):
    interval: float = field(
        default=1.0,
        metadata={"help": "Fixed interval for the fixed distribution."},
    )

    @classmethod
    def get_type(cls):
        return IntervalGeneratorType.FIXED
