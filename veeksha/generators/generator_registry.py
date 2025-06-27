from veeksha.types import RequestGeneratorType, SessionGeneratorType
from veeksha.types.base_registry import BaseRegistry

from .lmeval_generator import LMEvalRequestGenerator
from .trace_generator import TraceRequestGenerator
from .synthetic_generator import SyntheticRequestGenerator, SyntheticSessionGenerator

class RequestGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> RequestGeneratorType:
        return RequestGeneratorType.from_str(key_str)  # type: ignore

RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(RequestGeneratorType.TRACE, TraceRequestGenerator)
RequestGeneratorRegistry.register(RequestGeneratorType.LMEVAL, LMEvalRequestGenerator)

# -------------------------------------------------------------------------------------

class SessionGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> SessionGeneratorType:
        return SessionGeneratorType.from_str(key_str)  # type: ignore

SessionGeneratorRegistry.register(
    SessionGeneratorType.SYNTHETIC, SyntheticSessionGenerator
)