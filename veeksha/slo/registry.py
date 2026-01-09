from veeksha.slo.slo import ConstantSlo
from veeksha.types import SloType
from veeksha.types.base_registry import BaseRegistry


class SloRegistry(BaseRegistry):
    """Registry mapping SLO config types to runtime SLO evaluators."""

    @classmethod
    def get_key_from_str(cls, key_str: str) -> SloType:
        return SloType.from_str(key_str)  # type: ignore


SloRegistry.register(SloType.CONSTANT, ConstantSlo)
