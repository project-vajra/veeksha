from veeksha.types import SLOType
from veeksha.types.base_registry import BaseRegistry

from veeksha.capacity_search.slo import (
    ConstantSLO,
    TTFTPredictionMultiplierSLO,
    TTFTPredictionOffsetSLO,
    DeadlineSLO,
)


class SLORegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> SLOType:
        return SLOType.from_str(key_str)  # type: ignore


SLORegistry.register(SLOType.CONSTANT, ConstantSLO)
SLORegistry.register(SLOType.TTFT_PREDICTION_MULTIPLIER, TTFTPredictionMultiplierSLO)
SLORegistry.register(SLOType.TTFT_PREDICTION_OFFSET, TTFTPredictionOffsetSLO)
SLORegistry.register(SLOType.DEADLINE, DeadlineSLO) 