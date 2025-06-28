import pandas as pd

from veeksha.config.generators.session_generator.base_generator import BaseSessionGeneratorConfig
from veeksha.config.core.base_poly_config import BasePolyConfig

class BaseSessionGenerator(BasePolyConfig):
    def __init__(self, config: BaseSessionGeneratorConfig):
        self.config = config

    def generate_sessions(self, requests_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
        