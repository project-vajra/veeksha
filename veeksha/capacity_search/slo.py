from typing import List, Tuple

import numpy as np

from veeksha.config.slo import BaseSloConfig, ConstantSloConfig
from veeksha.logger import init_logger
from veeksha.metrics.metric_registry import MetricRegistry
from veeksha.metrics.metric_store import MetricStore

logger = init_logger(__name__)


class BaseSlo:
    """Base class for a single SLO definition."""

    def __init__(self, config: BaseSloConfig):
        self.config = config

    def get_threshold(self) -> float:
        """Get the threshold value for this SLO."""
        raise NotImplementedError

    def evaluate(
        self,
        metric_store: MetricStore,
    ) -> Tuple[bool, float]:
        """Evaluate this SLO against request metrics."""
        raise NotImplementedError

    def get_slo_metric_key(self) -> str:
        """Get the metric key for this SLO."""
        raise NotImplementedError


class ConstantSlo(BaseSlo):
    """SLO with a fixed constant value threshold."""

    def __init__(self, config: ConstantSloConfig):
        super().__init__(config)
        self.config: ConstantSloConfig = config  # Type annotation for clarity

    def evaluate(
        self,
        metric_store: MetricStore,
    ) -> Tuple[bool, float]:
        """Evaluate this simple metric SLO."""
        values = self._extract_metric_values(metric_store)
        if not values:
            logger.warning(f"No values found for metric {self.config.metric}")
            return False, float("inf")

        # Calculate percentile
        metric_value = float(np.percentile(values, self.config.percentile * 100))
        threshold = self.get_threshold()

        return metric_value <= threshold, metric_value

    def _extract_metric_values(self, metric_store: MetricStore) -> List[float]:
        """Extract metric values from the in-memory metric store."""
        spec = MetricRegistry.get(self.config.metric)
        # Use request-level metrics as the source of values
        return spec.extract(metric_store.request_level_metrics.to_dict())

    def get_threshold(self) -> float:
        return self.config.value

    def __str__(self) -> str:
        return f"ConstantSlo(metric={self.config.metric}, p{self.config.percentile*100:.0f} <= {self.config.value})"

    def get_slo_metric_key(self) -> str:
        return f"{self.config.metric}_p{self.config.percentile*100:.0f}"


class SloSet:
    """Set of SLOs for a benchmark to meet."""

    def __init__(self, slos: List[BaseSloConfig]):
        from veeksha.capacity_search.slo_registry import SloRegistry

        self.slos = [
            SloRegistry.get(slo_config.get_type(), config=slo_config)
            for slo_config in slos
        ]

    def __str__(self) -> str:
        if not self.slos:
            return "SloSet(empty)"

        slo_descriptions = []
        for i, slo in enumerate(self.slos, 1):
            slo_descriptions.append(f"  {i}. {str(slo)}")

        return f"SloSet({len(self.slos)} SLOs):\n" + "\n".join(slo_descriptions)
