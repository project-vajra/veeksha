import numpy as np

from veeksha.new.config.generator.length import (
    UniformLengthGeneratorConfig,
)
from veeksha.new.generator.length.base import (
    BaseLengthGenerator,
)


class UniformLengthGenerator(BaseLengthGenerator):
    def __init__(
        self, config: UniformLengthGeneratorConfig, rng: np.random.RandomState
    ):
        self.config = config
        self.rng = rng

    def get_next_length(self) -> int:
        return int(self.rng.uniform(self.config.min_length, self.config.max_length))
