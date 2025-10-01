import random
from typing import Tuple

from veeksha.config.generators.length_generator.uniform_generator import (
    UniformRequestLengthGeneratorConfig,
)
from veeksha.generators.length_generator.base_generator import (
    BaseRequestLengthGenerator,
)


class UniformRequestLengthGenerator(BaseRequestLengthGenerator):
    def __init__(self, config: UniformRequestLengthGeneratorConfig, rng: random.Random):
        self.config = config
        self.rng = rng

    def get_next_num_tokens(self) -> Tuple[float, float]:
        total_tokens = self.rng.uniform(
            self.config.min_tokens,
            self.config.max_tokens,
        )

        decode_tokens = total_tokens / (1 + self.config.prefill_to_decode_ratio)
        prefill_tokens = total_tokens - decode_tokens

        return prefill_tokens, decode_tokens
