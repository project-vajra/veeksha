from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class BaseRequestGeneratorConfig(BasePolyConfig):
    max_tokens: int = field(
        default=8192, metadata={"help": "Maximum number of tokens allowed."}
    )
