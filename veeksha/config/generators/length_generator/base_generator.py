from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.constants.configuration_constants import DEFAULT_SEED

@frozen_dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=DEFAULT_SEED, metadata={"help": "Random seed for the request length generator."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens allowed in a request."}
    )