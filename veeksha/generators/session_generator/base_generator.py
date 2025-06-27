from typing import List

from veeksha.config.generators.session_generator.base_generator import BaseSessionGeneratorConfig
from veeksha.core.request_config import RequestConfig
from veeksha.config.core.base_poly_config import BasePolyConfig

class BaseSessionGenerator(BasePolyConfig):
    def __init__(self, config: BaseSessionGeneratorConfig):
        self.config = config

    def get_session(self) -> List[RequestConfig]:
        raise NotImplementedError
        