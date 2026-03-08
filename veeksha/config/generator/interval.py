from vidhi import BasePolyConfig, field, frozen_dataclass

from veeksha.types import IntervalGeneratorType


@frozen_dataclass
class BaseIntervalGeneratorConfig(BasePolyConfig):
    """Wait-time sampling strategy (poisson, gamma, or fixed)."""


@frozen_dataclass
class GammaIntervalGeneratorConfig(BaseIntervalGeneratorConfig):
    arrival_rate: float = field(
        1.0, help="Arrival rate for the Gamma distribution."
    )
    cv: float = field(
        0.5, help="Coefficient of variation for the Gamma distribution."
    )

    @classmethod
    def get_type(cls):
        return IntervalGeneratorType.GAMMA


@frozen_dataclass
class PoissonIntervalGeneratorConfig(BaseIntervalGeneratorConfig):
    arrival_rate: float = field(
        1.0, help="Arrival rate for the Poisson distribution."
    )

    @classmethod
    def get_type(cls):
        return IntervalGeneratorType.POISSON


@frozen_dataclass
class FixedIntervalGeneratorConfig(BaseIntervalGeneratorConfig):
    interval: float = field(
        1.0, help="Fixed interval for the fixed distribution."
    )

    @classmethod
    def get_type(cls):
        return IntervalGeneratorType.FIXED
