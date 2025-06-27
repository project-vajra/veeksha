from typing import Tuple

from veeksha.config.generators.length_generator.fixed_generator import FixedRequestLengthGeneratorConfig
from veeksha.generators.length_generator.base_generator import (
    BaseRequestLengthGenerator,
)


class FixedRequestLengthGenerator(BaseRequestLengthGenerator):
    def __init__(self, config: FixedRequestLengthGeneratorConfig):
        self.config = config

    def get_next_num_tokens(self) -> Tuple[float, float]:
        return (
            self.config.prefill_tokens,
            self.config.decode_tokens,
        )
