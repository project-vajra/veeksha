from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum number of tokens allowed in a request."},
    )
