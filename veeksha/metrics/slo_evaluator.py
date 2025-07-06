"""SLO Evaluator for flexible SLO evaluation."""

import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from veeksha.config.slo import SLOSet, BaseSLO, SLOMetric
from veeksha.logger import init_logger

logger = init_logger(__name__)


class SLOEvaluator:
    """Evaluates SLOs against request-level metrics."""
    
    def __init__(self, 
                 slo_set: SLOSet,
                 predictions: Optional[Dict[int, float]] = None):
        """Initialize SLO evaluator.
        
        Args:
            slo_config: Composable SLO configuration
            predictions: Optional predictions for dynamic SLOs
        """
        self.slo_set = slo_set
        self.predictions = predictions or {}
        
    def evaluate_request_metrics(self, 
                               request_metrics_file: str) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate SLOs against request-level metrics.
        
        Args:
            request_metrics_file: Path to request-level metrics JSON file
            
        Returns:
            Tuple of (is_under_sla, metrics_dict) where:
                - is_under_sla: True if SLOs are met based on composition logic
                - metrics_dict: Dictionary of evaluated metrics and their values
        """
        with open(request_metrics_file, "r") as f:
            request_level_metrics = json.load(f)
        
        slo_results: List[bool] = []
        metrics_dict: Dict[str, Any] = {}
        
        for slo in self.slo_set.slos:
            is_met, metric_value, threshold = self._evaluate_single_slo(slo, request_level_metrics)
            slo_results.append(is_met)
            
            # Store metric value with descriptive key
            metric_key = f"{slo.metric.value}_p{int(slo.percentile * 100)}"
            if slo.name:
                metric_key = f"{slo.name.replace(' ', '_')}_{metric_key}"
            metrics_dict[metric_key] = metric_value
            
            # Log individual SLO result
            logger.debug(f"SLO '{slo.name or slo.metric.value}' "
                        f"(P{slo.percentile * 100}): "
                        f"{'MET' if is_met else 'MISSED'} "
                        f"(value={metric_value:.4f}, threshold={threshold:.4f})")
        
        # Apply composition logic
        if self.slo_set.require_all:
            is_under_sla = all(slo_results)
        else:
            is_under_sla = any(slo_results)
            
        return is_under_sla, metrics_dict
    
    def _evaluate_single_slo(self, 
                           slo: BaseSLO, 
                           request_metrics: Dict[str, Any]) -> Tuple[bool, float, float]:
        """Evaluate a single SLO.
        
        Args:
            slo: SLO definition to evaluate
            request_metrics: Request-level metrics dictionary
            
        Returns:
            Tuple of (is_met, metric_value, threshold)
        """
        metric_values = self._extract_metric_values(slo.metric, request_metrics)
        
        if not metric_values:
            logger.warning(f"No values found for metric {slo.metric.value}")
            return False, 0.0, 0.0

        # Handle prediction-based SLOs which are evaluated per-request
        if hasattr(slo, "predictor_field"):
            per_request_met: List[bool] = []
            predictor_field = getattr(slo, "predictor_field")
            num_tokens_list = request_metrics.get(predictor_field, [])
            
            for i, num_tokens in enumerate(num_tokens_list):
                request_threshold = slo.get_threshold(
                    predictions=self.predictions,
                    request_value=num_tokens
                )
                request_metric_value = self._get_request_metric_value(
                    slo.metric, request_metrics, i
                )
                if request_metric_value is not None:
                    per_request_met.append(request_metric_value <= request_threshold)

            fraction_met = (
                sum(per_request_met) / len(per_request_met) if per_request_met else 0.0
            )
            # For prediction-based SLOs, we check if the fraction of requests
            # meeting the SLO is >= the percentile.
            is_met = fraction_met >= slo.percentile
            
            # For logging purposes, we still report the aggregate percentile of the metric
            metric_value = np.quantile(metric_values, slo.percentile)
            # The threshold is dynamic, so for logging we just use the base value (offset/multiplier)
            threshold = getattr(slo, "value", 0.0)
            return is_met, metric_value, threshold

        # Handle constant SLOs by comparing the percentile value
        else:
            metric_value = np.quantile(metric_values, slo.percentile)
            threshold = slo.get_threshold()
            is_met = metric_value <= threshold
            return is_met, metric_value, threshold
    
    def _extract_metric_values(self, 
                             metric: SLOMetric, 
                             request_metrics: Dict[str, Any]) -> List[float]:
        """Extract metric values from request metrics."""
        values = request_metrics.get(metric.value, [])
        if metric == SLOMetric.TBT and values and isinstance(values[0], list):
            # Flatten the list of lists for TBT
            return [item for sublist in values for item in sublist]
        return values
    
    def _get_request_metric_value(self,
                                metric: SLOMetric,
                                request_metrics: Dict[str, Any],
                                request_idx: int) -> Optional[float]:
        """Get metric value for a specific request."""
        values = request_metrics.get(metric.value)
        if values and request_idx < len(values):
            return values[request_idx]
        return None
    
    def get_metrics_summary(self, metrics_dict: Dict[str, Any]) -> str:
        """Get a human-readable summary of metrics."""
        summary_parts = []
        for key, value in metrics_dict.items():
            summary_parts.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
        return " - ".join(summary_parts)