from veeksha.types.base_registry import BaseRegistry
from veeksha.types.request_length_generator_type import RequestLengthGeneratorType

from .fixed_request_length_generator import FixedRequestLengthGenerator
from .trace_request_length_generator import TraceRequestLengthGenerator
from .uniform_request_length_generator import UniformRequestLengthGenerator
from .zipf_request_length_generator import ZipfRequestLengthGenerator


class RequestLengthGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> RequestLengthGeneratorType:
        return RequestLengthGeneratorType.from_str(key_str)  # type: ignore


RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.ZIPF, ZipfRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.UNIFORM, UniformRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.TRACE, TraceRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.FIXED, FixedRequestLengthGenerator
)
