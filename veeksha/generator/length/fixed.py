from veeksha.config.generator.length import (
    FixedLengthGeneratorConfig,
)
from veeksha.generator.length.base import (
    BaseLengthGenerator,
)


class FixedLengthGenerator(BaseLengthGenerator):
    def __init__(self, config: FixedLengthGeneratorConfig, rng=None):
        self.config = config
        self.rng = rng

    def get_next_value(self) -> int:
        return self.config.value
