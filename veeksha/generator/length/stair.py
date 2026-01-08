from __future__ import annotations

from veeksha.config.generator.length import StairLengthGeneratorConfig
from veeksha.generator.length.base import BaseLengthGenerator


class StairLengthGenerator(BaseLengthGenerator):
    """Deterministic stair-step generator.

    Emits the configured values in-order, repeating each step `repeat_each` times.
    When `wrap` is enabled, cycles back to the beginning after the last step.
    """

    def __init__(self, config: StairLengthGeneratorConfig, rng=None):
        self.config = config
        self.rng = rng

        self._step_idx = 0
        self._repeat_idx = 0

    def get_next_value(self) -> int:
        value = self.config.values[self._step_idx]

        self._repeat_idx += 1
        if self._repeat_idx >= self.config.repeat_each:
            self._repeat_idx = 0
            self._step_idx += 1
            if self._step_idx >= len(self.config.values):
                self._step_idx = 0 if self.config.wrap else len(self.config.values) - 1

        return int(value)
