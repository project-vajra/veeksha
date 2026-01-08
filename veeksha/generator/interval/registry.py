from veeksha.types import IntervalGeneratorType
from veeksha.types.base_registry import BaseRegistry

from .fixed import FixedIntervalGenerator
from .gamma import GammaIntervalGenerator
from .poisson import PoissonIntervalGenerator


class IntervalGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> IntervalGeneratorType:
        return IntervalGeneratorType.from_str(key_str)  # type: ignore


IntervalGeneratorRegistry.register(IntervalGeneratorType.GAMMA, GammaIntervalGenerator)
IntervalGeneratorRegistry.register(
    IntervalGeneratorType.POISSON, PoissonIntervalGenerator
)
IntervalGeneratorRegistry.register(IntervalGeneratorType.FIXED, FixedIntervalGenerator)
