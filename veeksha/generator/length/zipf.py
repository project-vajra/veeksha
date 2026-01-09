from typing import Optional

import numpy as np

from veeksha.config.generator.length import (
    ZipfLengthGeneratorConfig,
)
from veeksha.generator.length.base import (
    BaseLengthGenerator,
)

ZIPF_LENGTH_GENERATOR_EPS = 1e-8


class ZipfGenerator:
    def __init__(
        self,
        min: int,
        max: int,
        theta: float,
        scramble: bool,
        rng: np.random.RandomState,
        scramble_seed: Optional[int],
    ) -> None:
        self.min = min
        self.max = max
        self.items = max - min + 1
        self.theta = theta
        self.zeta_2 = self._zeta(2, self.theta)
        self.alpha = 1.0 / (1.0 - self.theta + ZIPF_LENGTH_GENERATOR_EPS)
        self.zetan = self._zeta(self.items, self.theta)
        self.eta = (1 - np.power(2.0 / self.items, 1 - self.theta)) / (
            1 - self.zeta_2 / (self.zetan + ZIPF_LENGTH_GENERATOR_EPS)
        )
        self.scramble = scramble
        self.rng = rng
        self.scramble_seed = scramble_seed

    def _zeta(self, count: float, theta: float) -> float:
        return np.sum(1 / (np.power(np.arange(1, count), theta)))

    def _next(self) -> int:
        u = self.rng.random_sample()
        uz = u * self.zetan

        if uz < 1.0:
            return self.min

        if uz < 1.0 + np.power(0.5, self.theta):
            return self.min + 1

        return self.min + int(
            (self.items) * np.power(self.eta * u - self.eta + 1, self.alpha)
        )

    def next(self) -> int:
        retval = self._next()
        if self.scramble and self.scramble_seed is not None:
            retval = self.min + hash((retval, self.scramble_seed)) % self.items

        return retval


class ZipfLengthGenerator(BaseLengthGenerator):
    def __init__(
        self,
        config: ZipfLengthGeneratorConfig,
        rng: np.random.RandomState,
    ):
        self.config = config
        self.rng = rng

        scramble_seed: Optional[int] = None
        if self.config.scramble:
            scramble_seed = int(rng.randint(0, 2**32 - 1))

        self._zipf_generator = ZipfGenerator(
            self.config.min,
            self.config.max,
            self.config.theta,
            self.config.scramble,
            self.rng,
            scramble_seed,
        )

    def get_next_value(self) -> int:
        return self._zipf_generator.next()
