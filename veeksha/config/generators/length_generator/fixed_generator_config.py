from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.length_generator.base_generator_config import (
    BaseRequestLengthGeneratorConfig,
)
from veeksha.types import RequestLengthGeneratorType


@frozen_dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = field(
        default=4096, metadata={"help": "Number of prefill tokens."}
    )
    decode_tokens: int = field(
        default=512, metadata={"help": "Number of decode tokens."}
    )

    @classmethod
    def get_type(cls) -> RequestLengthGeneratorType:
        return RequestLengthGeneratorType.FIXED
