from veeksha.types import SessionGeneratorType
from veeksha.types.base_registry import BaseRegistry

from .trace_synthetic_generator import TraceSyntheticSessionGenerator

class SessionGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> SessionGeneratorType:
        return SessionGeneratorType.from_str(key_str)  # type: ignore

SessionGeneratorRegistry.register(
    SessionGeneratorType.TRACE_SYNTHETIC, TraceSyntheticSessionGenerator
)