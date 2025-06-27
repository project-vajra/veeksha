from veeksha.types import RequestGeneratorType
from veeksha.types.base_registry import BaseRegistry

from .lmeval_generator import LMEvalRequestGenerator
from .trace_generator import TraceRequestGenerator
from .synthetic_generator import SyntheticRequestGenerator

class RequestGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> RequestGeneratorType:
        return RequestGeneratorType.from_str(key_str)  # type: ignore

RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(RequestGeneratorType.TRACE, TraceRequestGenerator)
RequestGeneratorRegistry.register(RequestGeneratorType.LMEVAL, LMEvalRequestGenerator)