from dataclasses import field
from enum import Enum
from typing import Any, Dict, List, Optional, Literal, Tuple
import numpy as np
import yaml

from veeksha.logger import init_logger
from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass

logger = init_logger(__name__)


class SLOMetric(str, Enum):
    """Available metrics for SLO evaluation."""

    TTFT = "ttft"
    TBT = "tbt"
    TPOT = "tpot"


@frozen_dataclass
class BaseSLO(BasePolyConfig):
    """Base class for a single SLO definition."""

    percentile: float = field(
        default=0.99,
        metadata={"help": "Percentile at which to evaluate the SLO (0.0-1.0)"},
    )
    name: Optional[str] = field(
        default=None, metadata={"help": "Human-readable name for this SLO"}
    )

    def __post_init__(self):
        """Validate SLO definition."""
        if not 0.0 <= self.percentile <= 1.0:
            raise ValueError(
                f"percentile must be between 0.0 and 1.0, got {self.percentile}"
            )

    def get_threshold(self, **kwargs: Any) -> float:
        """Get the threshold value for this SLO."""
        raise NotImplementedError
    
    def evaluate(self, request_metrics: Dict[str, Any], predictions: Optional[Dict[int, float]] = None) -> Tuple[bool, float]:
        """Evaluate this SLO against request metrics."""
        raise NotImplementedError


@frozen_dataclass
class SimpleMetricSLO(BaseSLO):
    """Base class for SLOs that evaluate a single metric."""
    
    metric: SLOMetric = field(default=SLOMetric.TTFT, metadata={"help": "The metric this SLO applies to"})
    
    def _extract_metric_values(self, request_metrics: Dict[str, Any]) -> List[float]:
        """Extract metric values from request metrics."""
        values = request_metrics.get(self.metric.value, [])
        if self.metric == SLOMetric.TBT and values and isinstance(values[0], list):
            # flatten the list of lists for TBT
            return [item for sublist in values for item in sublist]
        return values
    
    def evaluate(self, request_metrics: Dict[str, Any], predictions: Optional[Dict[int, float]] = None) -> Tuple[bool, float]:
        """Evaluate this simple metric SLO."""
        values = self._extract_metric_values(request_metrics)
        if not values:
            logger.warning(f"No values found for metric {self.metric.value}")
            return False, float('inf')
        
        # Calculate percentile
        metric_value = float(np.percentile(values, self.percentile * 100))
        threshold = self.get_threshold(predictions=predictions, request_metrics=request_metrics)
        
        return metric_value <= threshold, metric_value
    
    @classmethod
    def get_type(cls) -> str:
        return "simple_metric"


@frozen_dataclass
class ConstantSLO(SimpleMetricSLO):
    """SLO with a fixed constant value threshold."""

    value: float = field(
        default=-1.0,
        metadata={
            "help": "The constant value for the SLO. If a percentage, from 0 to 1. If a time, in seconds."
        }
    )

    def __post_init__(self):
        """Validate SLO definition."""
        super().__post_init__()
        if self.value <= 0.0:
            raise ValueError("ConstantSLO: value must be specified and must be > 0")

    @classmethod
    def get_type(cls) -> str:
        return "constant"

    def get_threshold(self, **kwargs: Any) -> float:
        return self.value


@frozen_dataclass
class PredictionBasedSLO(SimpleMetricSLO):
    """Base class for SLOs based on predictions."""
    
    metric: Literal[SLOMetric.TTFT] = field(
        default=SLOMetric.TTFT,
        init=False,
        metadata={"help": "The metric this SLO applies to. Always TTFT."},
    )
    predictor_field: str = field(
        default="num_total_tokens",
        metadata={"help": "Field name to use for prediction lookup"},
    )
    min_value: Optional[float] = field(
        default=None, metadata={"help": "Minimum value for the SLO (for clamping)"}
    )
    max_value: Optional[float] = field(
        default=None, metadata={"help": "Maximum value for the SLO (for clamping)"}
    )

    @classmethod
    def get_type(cls) -> str:
        return "prediction_based"

    def _get_clamped_threshold(self, threshold: float) -> float:
        """Apply min/max clamping to the threshold."""
        if self.min_value is not None:
            threshold = max(threshold, self.min_value)
        if self.max_value is not None:
            threshold = min(threshold, self.max_value)
        return threshold

    def get_threshold(self, **kwargs: Any) -> float:
        """Get threshold for prediction-based SLO."""
        predictions = kwargs.get('predictions')
        request_metrics = kwargs.get('request_metrics')
        
        if not predictions or not request_metrics:
            raise ValueError("Prediction-based SLOs require both predictions and request_metrics")
        
        # Get the predictor field value (e.g., num_total_tokens)
        predictor_values = request_metrics.get(self.predictor_field, [])
        if not predictor_values:
            raise ValueError(f"No values found for predictor field {self.predictor_field}")
        
        # For now, use the first value - this could be extended to handle multiple values
        request_value = predictor_values[0] if isinstance(predictor_values, list) else predictor_values
        return self._calculate_threshold(predictions, request_value)
    
    def _calculate_threshold(self, predictions: Dict[int, float], request_value: float) -> float:
        """Calculate threshold based on prediction. To be implemented by subclasses."""
        raise NotImplementedError


@frozen_dataclass
class TTFTPredictionMultiplierSLO(PredictionBasedSLO):
    """SLO threshold is a multiplier of a predicted TTFT value."""

    value: float = field(
        default=-1.0,
        metadata={"help": "The multiplier for the SLO"},
    )

    @classmethod
    def get_type(cls) -> str:
        return "ttft_prediction_multiplier"

    def _calculate_threshold(self, predictions: Dict[int, float], request_value: float) -> float:
        """Calculate threshold based on prediction multiplier."""
        base_prediction = predictions.get(int(request_value), 0.0)
        threshold = base_prediction * self.value
        return self._get_clamped_threshold(threshold)
    
    def __post_init__(self):
        """Validate SLO definition."""
        super().__post_init__()
        raise NotImplementedError("TTFTPredictionMultiplierSLO is not implemented")
        if self.value <= 0.0:
            raise ValueError("TTFTPredictionMultiplierSLO: value must be specified and must be > 0")


@frozen_dataclass
class TTFTPredictionOffsetSLO(PredictionBasedSLO):
    """SLO threshold is a predicted TTFT value plus an offset."""

    value: float = field(
        default=-1.0,
        metadata={"help": "The offset (slack) for the SLO"},
    )

    @classmethod
    def get_type(cls) -> str:
        return "ttft_prediction_offset"

    def _calculate_threshold(self, predictions: Dict[int, float], request_value: float) -> float:
        """Calculate threshold based on prediction offset."""
        base_prediction = predictions.get(int(request_value), 0.0)
        threshold = base_prediction + self.value
        return self._get_clamped_threshold(threshold)
    
    def __post_init__(self):
        """Validate SLO definition."""
        super().__post_init__()
        raise NotImplementedError("TTFTPredictionOffsetSLO is not implemented")
        if self.value < 0.0:
            raise ValueError("TTFTPredictionOffsetSLO: value must be specified and must be >= 0")


@frozen_dataclass
class DeadlineSLO(BaseSLO):
    """SLO that evaluates deadline miss rate based on both TTFT and TBT thresholds."""
    
    # todo check bounds for fluidity index target threshold
    
    ttft_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "Fixed TTFT threshold in seconds. If None, uses prediction-based threshold."}
    )
    tbt_threshold: float = field(
        default=-1.0,
        metadata={"help": "TBT threshold in seconds"}
    )
    ttft_prediction_type: Optional[Literal["offset", "multiplier"]] = field(
        default=None,
        metadata={"help": "Type of prediction-based TTFT threshold. If None, uses ttft_threshold."}
    )
    ttft_prediction_value: Optional[float] = field(
        default=None,
        metadata={"help": "Value for prediction-based TTFT threshold (offset or multiplier)"}
    )
    ttft_min_value: Optional[float] = field(
        default=None, metadata={"help": "Minimum value for TTFT threshold (for clamping)"}
    )
    ttft_max_value: Optional[float] = field(
        default=None, metadata={"help": "Maximum value for TTFT threshold (for clamping)"}
    )
    
    def __post_init__(self):
        """Validate DeadlineSLO definition."""
        super().__post_init__()
        
        if self.ttft_prediction_type is not None:
            raise NotImplementedError("Prediction-based TTFT threshold is not implemented")
        
        if self.ttft_threshold is None and self.ttft_prediction_type is None:
            raise ValueError("DeadlineSLO: Must specify either ttft_threshold or ttft_prediction_type")
        
        if self.ttft_threshold is not None and self.ttft_prediction_type is not None:
            raise ValueError("DeadlineSLO: Cannot specify both ttft_threshold and ttft_prediction_type")
        
        if self.ttft_prediction_type is not None and self.ttft_prediction_value is None:
            raise ValueError("DeadlineSLO: Must specify ttft_prediction_value when using ttft_prediction_type")
        
        if self.ttft_prediction_type == "offset" and self.ttft_prediction_value < 0:
            raise ValueError("DeadlineSLO: ttft_prediction_value must be >= 0 for offset type")
        
        if self.ttft_prediction_type == "multiplier" and self.ttft_prediction_value <= 0:
            raise ValueError("DeadlineSLO: ttft_prediction_value must be > 0 for multiplier type")
        
        if self.tbt_threshold <= 0:
            raise ValueError("DeadlineSLO: tbt_threshold must be specified and > 0")

    @classmethod
    def get_type(cls) -> str:
        return "deadline"

    def get_threshold(self, **kwargs: Any) -> float:
        """For deadline SLO, threshold is the deadline miss rate threshold (not used in typical evaluation)."""
        # This is not typically used since deadline evaluation is different
        return 0.0
    
    def _get_ttft_threshold(self, predictions: Optional[Dict[int, float]], request_metrics: Dict[str, Any]) -> float:
        """Get the TTFT threshold (either fixed or prediction-based)."""
        if self.ttft_threshold is not None:
            return self.ttft_threshold
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
            
            if any(tbt > self.tbt_threshold for tbt in request_tbt_values):
                missed_deadlines += 1
        
        deadline_miss_rate = missed_deadlines / total_requests if total_requests > 0 else 1.0
        
        return deadline_miss_rate <= self.percentile, deadline_miss_rate


@frozen_dataclass
class SLOSet:
    """Composable set of SLOs for a benchmark to meet."""

    slos: List[BaseSLO] = field(
        default_factory=list,
        metadata={"help": "List of SLO definitions to evaluate"},
    )
    require_all: bool = field(
        default=True,
        metadata={"help": "If True, all SLOs must be met. If False, any SLO can be met."},
    )


def get_slos_from_config(slos_config_file: str) -> SLOSet:
    """Get or create SLOSet from a config file."""

    # Load from external file
    if slos_config_file.endswith(
        ".json"
    ) or slos_config_file.endswith(".yaml"):
        with open(slos_config_file, "r") as f:
            slo_dict = yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported config file format: {slos_config_file}"
        )

    # Handle polymorphic deserialization for slos
    slo_definitions = []
    for slo_def_dict in slo_dict.get("slos", []):
        slo_type_str = slo_def_dict.pop("type", None)
        if not slo_type_str:
            raise ValueError(
                "Each SLO definition in config file must have a 'type'"
            )
        from veeksha.capacity_search.slo_registry import SLORegistry
        slo_instance = SLORegistry.get_from_str(slo_type_str, **slo_def_dict)
        slo_definitions.append(slo_instance)
        
    slos = SLOSet(
        slos=slo_definitions, require_all=slo_dict.get("require_all", True)
    )
    logger.info(f"Loaded {len(slos.slos)} SLOs from {slos_config_file}")
    
    for slo in slos.slos:
        logger.info(f"SLO: {slo}")
    
    return slos