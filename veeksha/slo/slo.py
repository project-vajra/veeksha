from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from veeksha.config.slo import BaseSloConfig, ConstantSloConfig
from veeksha.logger import init_logger
from veeksha.slo.metrics import extract_metric_values

logger = init_logger(__name__)


class BaseSlo:
    """Base class for a single SLO definition."""

    def __init__(self, config: BaseSloConfig):
        self.config = config

    def get_threshold(self) -> float:
        """Return the threshold value for this SLO."""
        raise NotImplementedError

    def evaluate(self, request_level_metrics: Dict[str, Any]) -> Tuple[bool, float]:
        """Evaluate this SLO against request-level metrics.

        Args:
            request_level_metrics: Dict-of-lists representation of request-level
                metrics (as decoded from request_level_metrics.jsonl).

        Returns:
            Tuple of (met, observed_value).
        """
        raise NotImplementedError

    def get_slo_metric_key(self) -> str:
        """Return a stable key identifying this SLO's evaluated value."""
        raise NotImplementedError


class ConstantSlo(BaseSlo):
    """SLO with a fixed constant value threshold."""

    def __init__(self, config: ConstantSloConfig):
        super().__init__(config)
        self.config: ConstantSloConfig = config

    def get_threshold(self) -> float:
        return self.config.value

    def evaluate(self, request_level_metrics: Dict[str, Any]) -> Tuple[bool, float]:
        values = extract_metric_values(self.config.metric, request_level_metrics)
        if not values:
            logger.warning("No values found for metric %s", self.config.metric)
            return False, float("inf")

        metric_value = _percentile(values, self.config.percentile)
        threshold = self.get_threshold()

        # Currently, supported SLO metrics are latency-like (lower is better).
        if math.isclose(metric_value, threshold, rel_tol=1e-9, abs_tol=1e-12):
            met = True
        else:
            met = metric_value < threshold
        return met, metric_value

    def __str__(self) -> str:
        return (
            f"ConstantSlo(metric={self.config.metric}, "
            f"p{self.config.percentile*100:.0f} <= {self.config.value})"
        )

    def get_slo_metric_key(self) -> str:
        return f"{self.config.metric}_p{self.config.percentile*100:.0f}"


class SloSet:
    """Set of SLOs for a benchmark to meet."""

    def __init__(self, slos: List[BaseSloConfig]):
        from veeksha.slo.registry import SloRegistry

        self.slos = [SloRegistry.get(slo.get_type(), config=slo) for slo in slos]

    def __str__(self) -> str:
        if not self.slos:
            return "SloSet(empty)"

        slo_descriptions = []
        for i, slo in enumerate(self.slos, 1):
            slo_descriptions.append(f"  {i}. {str(slo)}")

        return f"SloSet({len(self.slos)} SLOs):\n" + "\n".join(slo_descriptions)


def _percentile(values: List[float], q: float) -> float:
    """Compute the qth percentile with linear interpolation.

    This matches NumPy's historical default behavior (`np.percentile(..., method="linear")`).

    Args:
        values: List of numeric values.
        q: Quantile in [0.0, 1.0].
    """
    if not values:
        return 0.0
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))

    xs = sorted(values)
    n = len(xs)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * w)
