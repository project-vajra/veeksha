from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.new.types import SessionGraphType


@frozen_dataclass
class BaseSessionGraphGeneratorConfig(BasePolyConfig):
    pass


@frozen_dataclass
class LinearSessionGraphGeneratorConfig(BaseSessionGraphGeneratorConfig):
    """
    Generator of linear request graphs (a sequence of requests).
    """

    # TODO: add generator for the number of requests
    num_request_generator = None

    @classmethod
    def get_type(cls):
        return SessionGraphType.LINEAR
