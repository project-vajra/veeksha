"""SLO Evaluator for flexible SLO evaluation."""

import json
from typing import Dict, List, Optional, Tuple, Any

from veeksha.capacity_search.slo import SloSet
from veeksha.logger import init_logger

logger = init_logger(__name__)


class SloEvaluator:
    """Evaluates SLOs in a SloSet against request-level metrics."""
    
    def __init__(self, 
                 slo_set: SloSet,
                 predictions: Optional[Dict[int, float]] = None):
        """Initialize SLO evaluator.
        
        Args:
            slo_set: SloSet containing SLOs to evaluate
            predictions: Optional predictions for dynamic SLOs
        """
        self.slo_set = slo_set
        self.predictions = predictions or {}
        
    def evaluate_slo_request_metrics(self, 
                                 request_metrics_file: str) -> Tuple[bool, Dict[str, float]]:
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
        metrics_dict: Dict[str, float] = {}
        
        for slo in self.slo_set.slos:
            result = slo.evaluate(request_level_metrics, self.predictions)
            slo_results.append(result[0])
            metrics_dict[slo.get_slo_metric_key()] = result[1]
            
            logger.info(f"SLO '{slo}' "
                        f"{'MET' if result[0] else 'MISSED'} "
                        f"(value={result[1]:.4f})")
        
        is_under_sla = all(slo_results)
            
        return is_under_sla, metrics_dict
    
    def get_metrics_summary(self, metrics_dict: Dict[str, Any]) -> str:
        """Get a human-readable summary of metrics."""
        summary_parts = []
        for key, value in metrics_dict.items():
            summary_parts.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
        return " - ".join(summary_parts)