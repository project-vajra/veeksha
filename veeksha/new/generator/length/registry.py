from veeksha.new.types import LengthGeneratorType
from veeksha.types.base_registry import BaseRegistry

from .fixed import FixedLengthGenerator
from .uniform import UniformLengthGenerator
from .zipf import ZipfLengthGenerator


class LengthGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> LengthGeneratorType:
        return LengthGeneratorType.from_str(key_str)  # type: ignore


LengthGeneratorRegistry.register(LengthGeneratorType.ZIPF, ZipfLengthGenerator)
LengthGeneratorRegistry.register(LengthGeneratorType.UNIFORM, UniformLengthGenerator)
LengthGeneratorRegistry.register(LengthGeneratorType.FIXED, FixedLengthGenerator)
