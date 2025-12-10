from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.length_generator.base_generator import (
    BaseRequestLengthGeneratorConfig,
)
from veeksha.types import RequestLengthGeneratorType


@frozen_dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = field(
        default=1024, metadata={"help": "Minimum number of tokens."}
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum number of input tokens allowed in a request."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0, metadata={"help": "Ratio of prefill tokens to decode tokens."}
    )

    @classmethod
    def get_type(cls) -> RequestLengthGeneratorType:
        return RequestLengthGeneratorType.UNIFORM

    def __post_init__(self):
        if self.prefill_to_decode_ratio <= 0:
            raise ValueError("prefill_to_decode_ratio must be > 0")
        if self.min_tokens <= 0:
            raise ValueError("min_tokens must be > 0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must be <= max_tokens")
