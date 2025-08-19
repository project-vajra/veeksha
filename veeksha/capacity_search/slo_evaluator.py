"""SLO Evaluator for flexible SLO evaluation."""

from typing import Any, Dict, List, Optional, Tuple

from veeksha.capacity_search.slo import SloSet
from veeksha.logger import init_logger
from veeksha.metrics.metric_store import MetricStore

logger = init_logger(__name__)


class SloEvaluator:
    """Evaluates SLOs in a SloSet against request-level metrics."""

    def __init__(self, slo_set: SloSet, predictions: Optional[Dict[int, float]] = None):
        """Initialize SLO evaluator.

        Args:
            slo_set: SloSet containing SLOs to evaluate
            predictions: Optional predictions for dynamic SLOs
        """
        self.slo_set = slo_set
        self.predictions = predictions or {}

    def evaluate_slo(self, metric_store: MetricStore) -> Tuple[bool, Dict[str, float]]:
        """Evaluate SLOs.

        The provided object must expose `.request_level_metrics.to_dict()`.
        """
        slo_results: List[bool] = []
        metrics_dict: Dict[str, float] = {}

        for slo in self.slo_set.slos:
            result = slo.evaluate(metric_store)
            slo_results.append(result[0])
            metrics_dict[slo.get_slo_metric_key()] = result[1]

            logger.info(
                f"SLO '{slo}' "
                f"{'MET' if result[0] else 'MISSED'} "
                f"(value={result[1]:.4f})"
            )

        is_under_sla = all(slo_results)
        logger.info(f"Is under SLA: {is_under_sla}")
        return is_under_sla, metrics_dict

    def get_metrics_summary(self, metrics_dict: Dict[str, Any]) -> str:
        """Get a human-readable summary of metrics."""
        summary_parts = []
        for key, value in metrics_dict.items():
            summary_parts.append(
                f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            )

        return " - ".join(summary_parts)
