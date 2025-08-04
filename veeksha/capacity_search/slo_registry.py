from veeksha.types import SloType
from veeksha.types.base_registry import BaseRegistry

from veeksha.capacity_search.slo import (
    ConstantSlo,
    TtftPredictionMultiplierSlo,
    TtftPredictionOffsetSlo,
    DeadlineSlo,
)


class SloRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> SloType:
        return SloType.from_str(key_str)  # type: ignore


SloRegistry.register(SloType.CONSTANT, ConstantSlo)
SloRegistry.register(SloType.TTFT_PREDICTION_MULTIPLIER, TtftPredictionMultiplierSlo)
SloRegistry.register(SloType.TTFT_PREDICTION_OFFSET, TtftPredictionOffsetSlo)
SloRegistry.register(SloType.DEADLINE, DeadlineSlo) 