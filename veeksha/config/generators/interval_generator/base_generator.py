from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.constants.configuration_constants import DEFAULT_SEED


@frozen_dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    pass
