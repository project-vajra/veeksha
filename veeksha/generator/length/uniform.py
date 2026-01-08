import numpy as np

from veeksha.config.generator.length import (
    UniformLengthGeneratorConfig,
)
from veeksha.generator.length.base import (
    BaseLengthGenerator,
)


class UniformLengthGenerator(BaseLengthGenerator):
    def __init__(
        self, config: UniformLengthGeneratorConfig, rng: np.random.RandomState
    ):
        self.config = config
        self.rng = rng

    def get_next_value(self) -> int:
        return int(self.rng.uniform(self.config.min, self.config.max))
