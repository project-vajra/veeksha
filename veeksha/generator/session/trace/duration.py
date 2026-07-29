"""Deterministic target-duration sampling shared by trace flavors."""

from __future__ import annotations

import random


def sample_clipped_gaussian_duration_s(
    target_s: float,
    spread_s: float | None,
    sigma_s: float | None,
    rng: random.Random,
) -> float:
    """Draw a duration from a symmetric clipped Gaussian."""
    if target_s <= 0:
        raise ValueError(f"target_s must be positive; got {target_s}")
    if spread_s is None:
        if sigma_s is not None:
            raise ValueError("sigma_s requires spread_s")
        return target_s
    if not 0 < spread_s < target_s:
        raise ValueError(
            f"spread_s must be in (0, target_s); got {spread_s} with {target_s}"
        )

    effective_sigma_s = spread_s / 2.0 if sigma_s is None else sigma_s
    if not 0 < effective_sigma_s <= spread_s:
        raise ValueError(
            "sigma_s must be in (0, spread_s]; "
            f"got {effective_sigma_s} with {spread_s}"
        )

    low_s = target_s - spread_s
    high_s = target_s + spread_s
    while True:
        duration_s = rng.gauss(target_s, effective_sigma_s)
        if low_s <= duration_s <= high_s:
            return duration_s
