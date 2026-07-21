"""Clipped-Gaussian target-duration sampling shared by trace flavors.

Both the audio (STT) and Seed-TTS-text (TTS) trace flavors let a run target a
per-session duration drawn from a clipped Gaussian centered on the median, so
soak / edge configs can request multi-minute sessions. The sampling logic is
factored here so the two flavors stay identical and deterministic.
"""

from __future__ import annotations

import random


def sample_clipped_gaussian_duration_s(
    target_s: float,
    spread_s: float | None,
    sigma_s: float | None,
    rng: random.Random,
) -> float:
    """Draw a per-session duration (seconds) from a clipped Gaussian.

    Returns ``target_s`` unchanged when ``spread_s`` is None. Otherwise draws
    ``Normal(target_s, sigma_s)`` (sigma defaults to ``spread_s / 2``) re-sampled
    until it lands inside ``[target_s - spread_s, target_s + spread_s]``. The
    clip is symmetric about the target, so the median stays at ``target_s`` and
    the bounds are hard limits.
    """
    if spread_s is None:
        return target_s
    if sigma_s is None:
        sigma_s = spread_s / 2.0
    low_s = target_s - spread_s
    high_s = target_s + spread_s
    while True:
        duration_s = rng.gauss(target_s, sigma_s)
        if low_s <= duration_s <= high_s:
            return duration_s
