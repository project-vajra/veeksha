from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.decorators import allow_from_file
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.constants.configuration_constants import DEFAULT_SEED


@allow_from_file
@frozen_dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Random seed for the request generator."},
    )
    max_tokens: int = field(
        default=8192, metadata={"help": "Maximum number of tokens allowed."}
    )
