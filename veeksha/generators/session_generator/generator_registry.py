from veeksha.types import SessionGeneratorType
from veeksha.types.base_registry import BaseRegistry

from .synthetic_generator import SyntheticSessionGenerator


class SessionGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> SessionGeneratorType:
        return SessionGeneratorType.from_str(key_str)  # type: ignore


SessionGeneratorRegistry.register(
    SessionGeneratorType.SYNTHETIC, SyntheticSessionGenerator
)
