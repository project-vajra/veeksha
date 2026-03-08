from vidhi import BasePolyConfig, field, frozen_dataclass

from veeksha.types import LengthGeneratorType


@frozen_dataclass
class BaseLengthGeneratorConfig(BasePolyConfig):
    """Length sampling strategy (fixed, uniform, zipf, or stair)."""


@frozen_dataclass
class FixedLengthGeneratorConfig(BaseLengthGeneratorConfig):
    value: int = field(8, help="Value to generate.")

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
        help="Ordered list of step values to emit.",
    )
    repeat_each: int = field(
        1, help="Number of consecutive emissions per step value before advancing."
    )
    wrap: bool = field(
        True,
        help="If True, cycle back to the first value after the last. "
        "If False, keep returning the last value.",
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
    min: int = field(6, help="Minimum value to generate.")
    max: int = field(12, help="Maximum value to generate.")

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
    theta: float = field(0.6, help="Theta parameter for the Zipf distribution.")
    scramble: bool = field(False, help="Whether to scramble the Zipf distribution.")
    min: int = field(6, help="Minimum value to generate.")
    max: int = field(12, help="Maximum value to generate.")

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


@frozen_dataclass
class InverseGaussianLengthGeneratorConfig(BaseLengthGeneratorConfig):
    mean: float = field(
        default=500.0,
        metadata={"help": "Mean parameter for the inverse Gaussian distribution."},
    )
    shape: float = field(
        default=300.0,
        metadata={
            "help": "Shape (lambda) parameter for the inverse Gaussian distribution. Lower values mean more spread."
        },
    )

    @classmethod
    def get_type(cls) -> LengthGeneratorType:
        return LengthGeneratorType.INVERSE_GAUSSIAN

    def __post_init__(self):
        if self.mean <= 0:
            raise ValueError("mean must be > 0")
        if self.shape <= 0:
            raise ValueError("shape must be > 0")
