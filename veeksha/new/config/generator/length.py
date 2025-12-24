from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.new.types import LengthGeneratorType


@frozen_dataclass
class BaseLengthGeneratorConfig(BasePolyConfig):
    pass


@frozen_dataclass
class FixedLengthGeneratorConfig(BaseLengthGeneratorConfig):
    value: int = field(default=4096, metadata={"help": "Value to generate."})

    @classmethod
    def get_type(cls) -> LengthGeneratorType:
        return LengthGeneratorType.FIXED


@frozen_dataclass
class UniformLengthGeneratorConfig(BaseLengthGeneratorConfig):
    min: int = field(default=1024, metadata={"help": "Minimum value to generate."})
    max: int = field(
        default=4096,
        metadata={"help": "Maximum value to generate."},
    )

    @classmethod
    def get_type(cls) -> LengthGeneratorType:
        return LengthGeneratorType.UNIFORM

    def __post_init__(self):
        if self.min <= 0:
            raise ValueError("min must be > 0")
        if self.max <= 0:
            raise ValueError("max must be > 0")
        if self.min > self.max:
            raise ValueError("min must be <= max")


@frozen_dataclass
class ZipfLengthGeneratorConfig(BaseLengthGeneratorConfig):
    theta: float = field(
        default=0.6, metadata={"help": "Theta parameter for the Zipf distribution."}
    )
    scramble: bool = field(
        default=False, metadata={"help": "Whether to scramble the Zipf distribution."}
    )
    min: int = field(default=1024, metadata={"help": "Minimum value to generate."})
    max: int = field(
        default=4096,
        metadata={"help": "Maximum value to generate."},
    )

    @classmethod
    def get_type(cls) -> LengthGeneratorType:
        return LengthGeneratorType.ZIPF

    def __post_init__(self):
        if self.min <= 0:
            raise ValueError("min must be > 0")
        if self.max <= 0:
            raise ValueError("max must be > 0")
        if self.min > self.max:
            raise ValueError("min must be <= max")
