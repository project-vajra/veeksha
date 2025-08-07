from dataclasses import field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml

from veeksha.logger import init_logger
from veeksha.config.slo import BaseSloConfig, ConstantSloConfig, TtftPredictionMultiplierSloConfig, TtftPredictionOffsetSloConfig, DeadlineSloConfig, PredictionBasedSloConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass

logger = init_logger(__name__)


class SloMetric(str, Enum):
    """Available metrics for SLO evaluation."""

    TTFT = "ttft"
    TBT = "tbt"
    TPOT = "tpot"


class BaseSlo:
    """Base class for a single SLO definition."""

    def __init__(self, config: BaseSloConfig):
        self.config = config

    def get_threshold(self, **kwargs: Any) -> float:
        """Get the threshold value for this SLO."""
        raise NotImplementedError
    
    def evaluate(self, request_metrics: Dict[str, Any], predictions: Optional[Dict[int, float]] = None) -> Tuple[bool, float]:
        """Evaluate this SLO against request metrics."""
        raise NotImplementedError


class SimpleMetricSlo:
    """Base class for SLOs that evaluate a single metric."""
    
    def _extract_metric_values(self, request_metrics: Dict[str, Any]) -> List[float]:
        """Extract metric values from request metrics."""
        values = request_metrics.get(self.config.metric, [])
        if self.config.metric == SloMetric.TBT and values and isinstance(values[0], list):
            # flatten the list of lists for TBT
            return [item for sublist in values for item in sublist]
        return values
    
    def evaluate(self, request_metrics: Dict[str, Any], predictions: Optional[Dict[int, float]] = None) -> Tuple[bool, float]:
        """Evaluate this simple metric SLO."""
        values = self._extract_metric_values(request_metrics)
        if not values:
            logger.warning(f"No values found for metric {self.metric}")
            return False, float('inf')
        
        # Calculate percentile
        metric_value = float(np.percentile(values, self.config.percentile * 100))
        threshold = self.get_threshold(predictions=predictions, request_metrics=request_metrics)
        
        return metric_value <= threshold, metric_value

class ConstantSlo(SimpleMetricSlo):
    """SLO with a fixed constant value threshold."""

    def __init__(self, config: ConstantSloConfig):
        self.config = config

    def get_threshold(self, **kwargs: Any) -> float:
        return self.config.value
    
    def __str__(self) -> str:
        return f"ConstantSlo(metric={self.config.metric}, p{self.config.percentile*100:.0f} <= {self.config.value})"


class PredictionBasedSlo(SimpleMetricSlo):
    """Base class for SLOs based on predictions."""

    def __init__(self, config: PredictionBasedSloConfig):
        self.config = config

    def _get_clamped_threshold(self, threshold: float) -> float:
        """Apply min/max clamping to the threshold."""
        if self.config.min_value is not None:
            threshold = max(threshold, self.config.min_value)
        if self.config.max_value is not None:
            threshold = min(threshold, self.config.max_value)
        return threshold

    def get_threshold(self, **kwargs: Any) -> float:
        """Get threshold for prediction-based SLO."""
        predictions = kwargs.get('predictions')
        request_metrics = kwargs.get('request_metrics')
        
        if not predictions or not request_metrics:
            raise ValueError("Prediction-based SLOs require both predictions and request_metrics")
        
        # Get the predictor field value (e.g., num_total_tokens)
        predictor_values = request_metrics.get(self.config.predictor_field, [])
        if not predictor_values:
            raise ValueError(f"No values found for predictor field {self.config.predictor_field}")
        
        # For now, use the first value - this could be extended to handle multiple values
        request_value = predictor_values[0] if isinstance(predictor_values, list) else predictor_values
        return self._calculate_threshold(predictions, request_value)
    
    def _calculate_threshold(self, predictions: Dict[int, float], request_value: float) -> float:
        """Calculate threshold based on prediction. To be implemented by subclasses."""
        raise NotImplementedError


class TtftPredictionMultiplierSlo(PredictionBasedSlo):
    """SLO threshold is a multiplier of a predicted TTFT value."""

    def __init__(self, config: TtftPredictionMultiplierSloConfig):
        self.config = config

    def _calculate_threshold(self, predictions: Dict[int, float], request_value: float) -> float:
        """Calculate threshold based on prediction multiplier."""
        base_prediction = predictions.get(int(request_value), 0.0)
        threshold = base_prediction * self.config.value
        return self._get_clamped_threshold(threshold)
    
    def __str__(self) -> str:
        bounds_str = ""
        if self.config.min_value is not None or self.config.max_value is not None:
            bounds_parts = []
            if self.config.min_value is not None:
                bounds_parts.append(f"min={self.config.min_value}")
            if self.config.max_value is not None:
                bounds_parts.append(f"max={self.config.max_value}")
            bounds_str = f", bounds=[{', '.join(bounds_parts)}]"
        return f"TtftPredictionMultiplierSlo(metric={self.config.metric}, p{self.config.percentile*100:.0f} <= {self.config.value}x * prediction[{self.config.predictor_field}]{bounds_str})"
    

class TtftPredictionOffsetSlo(PredictionBasedSlo):
    """SLO threshold is a predicted TTFT value plus an offset."""

    def __init__(self, config: TtftPredictionOffsetSloConfig):
        self.config = config

    def _calculate_threshold(self, predictions: Dict[int, float], request_value: float) -> float:
        """Calculate threshold based on prediction offset."""
        base_prediction = predictions.get(int(request_value), 0.0)
        threshold = base_prediction + self.config.value
        return self._get_clamped_threshold(threshold)
    
    def __str__(self) -> str:
        bounds_str = ""
        if self.config.min_value is not None or self.config.max_value is not None:
            bounds_parts = []
            if self.config.min_value is not None:
                bounds_parts.append(f"min={self.config.min_value}")
            if self.config.max_value is not None:
                bounds_parts.append(f"max={self.config.max_value}")
            bounds_str = f", bounds=[{', '.join(bounds_parts)}]"
        return f"TtftPredictionOffsetSlo(metric={self.config.metric}, p{self.config.percentile*100:.0f} <= prediction[{self.config.predictor_field}] + {self.config.value}{bounds_str})"


class DeadlineSlo:
    """SLO that evaluates deadline miss rate based on both TTFT and TBT thresholds."""
    
    # todo check bounds for fluidity index target threshold
    
    def __init__(self, config: DeadlineSloConfig):
        self.config = config

    def get_threshold(self, **kwargs: Any) -> float:
        """For deadline SLO, threshold is the deadline miss rate threshold (not used in typical evaluation)."""
        # This is not typically used since deadline evaluation is different
        return 0.0
    
    def __str__(self) -> str:
        ttft_desc = f"ttft <= {self.config.ttft_threshold}" if self.config.ttft_threshold is not None else "ttft <= prediction"
        return f"DeadlineSlo(deadline_miss_rate <= {self.config.percentile}, constraints=[{ttft_desc}, tbt <= {self.config.tbt_threshold}])"
    
    def _get_ttft_threshold(self, predictions: Optional[Dict[int, float]], request_metrics: Dict[str, Any]) -> float:
        """Get the TTFT threshold (either fixed or prediction-based)."""
        if self.config.ttft_threshold is not None:
            return self.config.ttft_threshold
        else:
            raise NotImplementedError("Prediction-based TTFT threshold is not implemented")
            
    def evaluate(self, request_metrics: Dict[str, Any], predictions: Optional[Dict[int, float]] = None) -> Tuple[bool, float]:
        """Evaluate deadline miss rate."""
        ttft_values = request_metrics.get("ttft", [])
        tbt_values = request_metrics.get("tbt", [])
        
        if not ttft_values or not tbt_values:
            logger.warning("DeadlineSLO: No values found for TTFT or TBT")
            return False, 1.0
        
        # Get TTFT threshold
        ttft_threshold = self._get_ttft_threshold(predictions, request_metrics)
        
        # Flatten TBT values if needed
        if tbt_values and isinstance(tbt_values[0], list):
            tbt_flat = [item for sublist in tbt_values for item in sublist]
        else:
            tbt_flat = tbt_values
        
        # Calculate deadline misses
        total_requests = len(ttft_values)
        missed_deadlines = 0
        
        for i, ttft in enumerate(ttft_values):
            # Check if TTFT exceeds threshold
            if ttft > ttft_threshold:
                missed_deadlines += 1
                continue
                
            # Check if any TBT value for this request exceeds threshold
            # Assume TBT values are grouped by request (this may need adjustment based on actual data structure)
            request_tbt_start = i * (len(tbt_flat) // total_requests) if total_requests > 0 else 0
            request_tbt_end = (i + 1) * (len(tbt_flat) // total_requests) if total_requests > 0 else len(tbt_flat)
            
            request_tbt_values = tbt_flat[request_tbt_start:request_tbt_end] if request_tbt_start < len(tbt_flat) else []
            
            if any(tbt > self.config.tbt_threshold for tbt in request_tbt_values):
                missed_deadlines += 1
        
        deadline_miss_rate = missed_deadlines / total_requests if total_requests > 0 else 1.0
        
        return deadline_miss_rate <= self.config.percentile, deadline_miss_rate


class SloSet:
    """Set of SLOs for a benchmark to meet."""
    
    def __init__(self, slos: List[BaseSloConfig]):
        from veeksha.capacity_search.slo_registry import SloRegistry
        self.slos = [SloRegistry.get(slo_config.get_type(), config=slo_config) for slo_config in slos]
    
    def __str__(self) -> str:
        if not self.slos:
            return "SloSet(empty)"
        
        slo_descriptions = []
        for i, slo in enumerate(self.slos, 1):
            slo_descriptions.append(f"  {i}. {str(slo)}")
        
        return f"SloSet({len(self.slos)} SLOs):\n" + "\n".join(slo_descriptions)
