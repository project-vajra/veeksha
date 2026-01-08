from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.types import LengthGeneratorType


@frozen_dataclass
class BaseLengthGeneratorConfig(BasePolyConfig):
    pass


@frozen_dataclass
class FixedLengthGeneratorConfig(BaseLengthGeneratorConfig):
    value: int = field(default=8, metadata={"help": "Value to generate."})

    @classmethod
    def get_type(cls) -> LengthGeneratorType:
        return LengthGeneratorType.FIXED


@frozen_dataclass
class StairLengthGeneratorConfig(BaseLengthGeneratorConfig):
    """Emits values in the provided order, optionally repeating each value a fixed
    number of times before stepping to the next.
    """

    values: list[int] = field(
        default_factory=lambda: [8, 16, 32, 64],
        metadata={"help": "Ordered list of step values to emit."},
    )
    repeat_each: int = field(
        default=1,
        metadata={
            "help": "Number of consecutive emissions per step value before advancing."
        },
    )
    wrap: bool = field(
        default=True,
        metadata={
            "help": "If True, cycle back to the first value after the last. "
            "If False, keep returning the last value."
        },
    )

    @classmethod
    def get_type(cls) -> LengthGeneratorType:
        return LengthGeneratorType.FIXED_STAIR

    def __post_init__(self):
        if not self.values:
            raise ValueError("values must be non-empty")
        if any(v <= 0 for v in self.values):
            raise ValueError("All values must be > 0")
        if self.repeat_each <= 0:
            raise ValueError("repeat_each must be > 0")


@frozen_dataclass
class UniformLengthGeneratorConfig(BaseLengthGeneratorConfig):
    min: int = field(default=6, metadata={"help": "Minimum value to generate."})
    max: int = field(
        default=12,
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
    min: int = field(default=6, metadata={"help": "Minimum value to generate."})
    max: int = field(
        default=12,
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
