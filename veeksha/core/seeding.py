from __future__ import annotations

import hashlib
import itertools
import random
from typing import Callable, TypeVar

import numpy as np

T = TypeVar("T")

_MAX_SEED = 2**32


def derive_seed(root_seed: int, *path: str) -> int:
    """Derive a deterministic child seed from a root seed and path components."""
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(root_seed.to_bytes(8, byteorder="big", signed=False))
    for component in path:
        hasher.update(component.encode("utf-8"))
    digest = int.from_bytes(hasher.digest(), byteorder="big", signed=False)
    return digest % _MAX_SEED


def random_from_seed(seed: int) -> random.Random:
    return random.Random(seed)


def numpy_random_from_seed(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _make_factory(
    root_seed: int,
    initializer: Callable[[int], T],
    path: tuple[str, ...],
) -> Callable[[], T]:
    counter = itertools.count()

    def factory() -> T:
        seed = derive_seed(root_seed, *path, str(next(counter)))
        return initializer(seed)

    return factory


def numpy_factory(root_seed: int, *path: str) -> Callable[[], np.random.RandomState]:
    # when something requires a np.random.RandomState, we use this function
    return _make_factory(root_seed, numpy_random_from_seed, path)


def random_factory(root_seed: int, *path: str) -> Callable[[], random.Random]:
    # when something requires a random.Random, we use this function
    return _make_factory(root_seed, random_from_seed, path)


def random_for_path(root_seed: int, *path: str) -> random.Random:
    return random_from_seed(derive_seed(root_seed, *path))


class SeedManager:
    """Hierarchical seed manager yielding deterministic RNG streams."""

    def __init__(self, seed: int):
        self._seed = seed % _MAX_SEED

    def child(self, *path: str) -> "SeedManager":
        return SeedManager(derive_seed(self._seed, *path))

    def random_factory(self, *path: str) -> Callable[[], random.Random]:
        return random_factory(self._seed, *path)

    def numpy_factory(self, *path: str) -> Callable[[], np.random.RandomState]:
        return numpy_factory(self._seed, *path)

    def random(self, *path: str) -> random.Random:
        return random_for_path(self._seed, *path)

    def numpy(self, *path: str) -> np.random.RandomState:
        return numpy_random_from_seed(derive_seed(self._seed, *path))

    @property
    def seed(self) -> int:
        return self._seed
