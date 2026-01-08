from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from veeksha.logger import init_logger
from veeksha.slo.metrics import lower_is_better
from veeksha.slo.slo import SloSet

logger = init_logger(__name__)


@dataclass(frozen=True)
class SloCheckResult:
    """Result for a single evaluated SLO."""

    met: bool
    slo_metric_key: str
    observed_value: float
    threshold: float
    percentile: float
    metric: Optional[str]
    name: Optional[str]
    lower_is_better: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "met": self.met,
            "slo_metric_key": self.slo_metric_key,
            "observed_value": self.observed_value,
            "threshold": self.threshold,
            "percentile": self.percentile,
            "metric": self.metric,
            "name": self.name,
            "lower_is_better": self.lower_is_better,
        }


@dataclass(frozen=True)
class SloEvaluationResult:
    """Aggregate SLO evaluation result for a benchmark run."""

    all_slos_met: bool
    results: List[SloCheckResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_slos_met": self.all_slos_met,
            "results": [r.to_dict() for r in self.results],
        }


class SloEvaluator:
    """Evaluates a SloSet against request-level metrics."""

    def __init__(self, slo_set: SloSet):
        self.slo_set = slo_set

    def evaluate(self, request_level_metrics: Dict[str, Any]) -> SloEvaluationResult:
        """Evaluate all SLOs.

        Returns:
            SloEvaluationResult with per-SLO details.
        """

        results: List[SloCheckResult] = []

        for slo in self.slo_set.slos:
            met, observed = slo.evaluate(request_level_metrics)
            metric_key = slo.get_slo_metric_key()

            threshold = slo.get_threshold()
            percentile = getattr(slo.config, "percentile", 1.0)
            metric = getattr(slo.config, "metric", None)
            name = getattr(slo.config, "name", None)

            lib = lower_is_better(metric) if isinstance(metric, str) else True

            results.append(
                SloCheckResult(
                    met=met,
                    slo_metric_key=metric_key,
                    observed_value=float(observed),
                    threshold=float(threshold),
                    percentile=float(percentile),
                    metric=metric,
                    name=name,
                    lower_is_better=lib,
                )
            )

            logger.info(
                "SLO '%s' %s (value=%.4f threshold=%.4f)",
                name or str(slo),
                "MET" if met else "MISSED",
                observed,
                threshold,
            )

        all_met = all(r.met for r in results) if results else True
        logger.info("All SLOs met: %s", all_met)
        return SloEvaluationResult(all_slos_met=all_met, results=results)
