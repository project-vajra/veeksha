import numpy as np

from veeksha.config.generator.length import (
    InverseGaussianLengthGeneratorConfig,
)
from veeksha.generator.length.base import (
    BaseLengthGenerator,
)


class InverseGaussianLengthGenerator(BaseLengthGenerator):
    def __init__(
        self,
        config: InverseGaussianLengthGeneratorConfig,
        rng: np.random.RandomState,
    ):
        self.config = config
        self.rng = rng

    def get_next_value(self) -> int:
        sample = float(self.rng.wald(self.config.mean, self.config.shape))
        return max(1, int(round(sample)))
