from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.constants.configuration_constants import DEFAULT_SEED


@frozen_dataclass
class BaseSessionGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=DEFAULT_SEED, metadata={"help": "Random seed for the session generator."}
    )