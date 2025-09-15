from veeksha.capacity_search.slo import ConstantSlo
from veeksha.types import SloType
from veeksha.types.base_registry import BaseRegistry


class SloRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> SloType:
        for member in SloType:
            if member.value == key_str:
                return member
        raise ValueError(f"No SloType with value '{key_str}'")


SloRegistry.register(SloType.CONSTANT, ConstantSlo)
