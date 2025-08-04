from dataclasses import field
from enum import Enum
from typing import Any, Dict, List, Optional, Literal, Tuple
import numpy as np
import yaml

from veeksha.logger import init_logger
from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass

from veeksha.types import SloType

logger = init_logger(__name__)


class SloMetric(str, Enum):
    """Available metrics for SLO evaluation."""

    TTFT = "ttft"
    TBT = "tbt"
    TPOT = "tpot"


@frozen_dataclass
class BaseSloConfig(BasePolyConfig):
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


@frozen_dataclass
class ConstantSloConfig(BaseSloConfig):
    """SLO with a fixed constant value threshold."""
    
    metric: SloMetric = field(default=SloMetric.TTFT, metadata={"help": "The metric this SLO applies to"})

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
        return SloType.CONSTANT


@frozen_dataclass
class PredictionBasedSloConfig:
    """Base class for SLOs based on predictions."""
    
    metric: SloMetric = field(
        default=SloMetric.TTFT,
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


@frozen_dataclass
class TtftPredictionMultiplierSloConfig(PredictionBasedSloConfig, BaseSloConfig):
    """SLO threshold is a multiplier of a predicted TTFT value."""

    value: float = field(
        default=-1.0,
        metadata={"help": "The multiplier for the SLO"},
    )

    @classmethod
    def get_type(cls) -> str:
        return SloType.TTFT_PREDICTION_MULTIPLIER

    def __post_init__(self):
        """Validate SLO definition."""
        super().__post_init__()
        raise NotImplementedError("TTFTPredictionMultiplierSLO is not implemented")
        if self.value <= 0.0:
            raise ValueError("TTFTPredictionMultiplierSLO: value must be specified and must be > 0")


@frozen_dataclass
class TtftPredictionOffsetSloConfig(PredictionBasedSloConfig, BaseSloConfig):
    """SLO threshold is a predicted TTFT value plus an offset."""

    value: float = field(
        default=-1.0,
        metadata={"help": "The offset (slack) for the SLO"},
    )

    @classmethod
    def get_type(cls) -> str:
        return SloType.TTFT_PREDICTION_OFFSET

    def __post_init__(self):
        """Validate SLO definition."""
        super().__post_init__()
        raise NotImplementedError("TTFTPredictionOffsetSLO is not implemented")
        if self.value < 0.0:
            raise ValueError("TTFTPredictionOffsetSLO: value must be specified and must be >= 0")


@frozen_dataclass
class DeadlineSloConfig(BaseSloConfig):
    """SLO that evaluates deadline miss rate based on both TTFT and TBT thresholds."""
    
    # todo check bounds for fluidity index target threshold
    
    ttft_threshold: Optional[float] = field(
        default=0.1,
        metadata={"help": "Fixed TTFT threshold in seconds. If None, uses prediction-based threshold."}
    )
    tbt_threshold: float = field(
        default=0.03,
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
        return SloType.DEADLINE


@frozen_dataclass
class SloSetConfig:
    """Composable set of SLOs for a benchmark to meet."""

    slos: List[BaseSloConfig] = field(
        default_factory=list,
        metadata={"help": "List of SLO definitions to evaluate"},
    )
    require_all: bool = field(
        default=True,
        metadata={"help": "If True, all SLOs must be met. If False, any SLO can be met."},
    )
