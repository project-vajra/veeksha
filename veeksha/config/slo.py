from abc import ABC
from dataclasses import field
from enum import Enum
from typing import Any, Dict, List, Optional, Literal

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass


class SLOMetric(str, Enum):
    """Available metrics for SLO evaluation."""

    TTFT = "ttft"
    TBT = "tbt"
    TPOT = "tpot"
    DEADLINE_MISS_RATE = "deadline_miss_rate"


@frozen_dataclass
class BaseSLO(BasePolyConfig):
    """Base class for a single SLO definition."""

    metric: SLOMetric = field(metadata={"help": "The metric this SLO applies to"})
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


@frozen_dataclass
class ConstantSLO(BaseSLO):
    """SLO with a fixed constant value threshold."""

    value: float = field(metadata={"help": "The constant value for the SLO"})

    @classmethod
    def get_type(cls) -> str:
        return "constant"

    def get_threshold(self, **kwargs: Any) -> float:
        return self.value


@frozen_dataclass
class PredictionSLO(BaseSLO):
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

    def _get_clamped_threshold(self, threshold: float) -> float:
        """Apply min/max clamping to the threshold."""
        if self.min_value is not None:
            threshold = max(threshold, self.min_value)
        if self.max_value is not None:
            threshold = min(threshold, self.max_value)
        return threshold


@frozen_dataclass
class PredictionMultiplierSLO(PredictionSLO):
    """SLO threshold is a multiplier of a predicted value."""

    value: float = field(metadata={"help": "The multiplier for the SLO"})

    @classmethod
    def get_type(cls) -> str:
        return "prediction_multiplier"

    def get_threshold(
        self, predictions: Dict[int, float], request_value: float, **kwargs: Any
    ) -> float:
        """Calculate threshold based on prediction multiplier."""
        base_prediction = predictions.get(int(request_value), 0.0)
        threshold = base_prediction * self.value
        return self._get_clamped_threshold(threshold)


@frozen_dataclass
class PredictionOffsetSLO(PredictionSLO):
    """SLO threshold is a predicted value plus an offset."""

    value: float = field(metadata={"help": "The offset (slack) for the SLO"})

    @classmethod
    def get_type(cls) -> str:
        return "prediction_offset"

    def get_threshold(
        self, predictions: Dict[int, float], request_value: float, **kwargs: Any
    ) -> float:
        """Calculate threshold based on prediction offset."""
        base_prediction = predictions.get(int(request_value), 0.0)
        threshold = base_prediction + self.value
        return self._get_clamped_threshold(threshold)


@frozen_dataclass
class SLOSet:
    """Composable set of SLOs."""

    slos: List[BaseSLO] = field(
        default_factory=list,
        metadata={"help": "List of SLO definitions to evaluate"},
    )
    require_all: bool = field(
        default=True,
        metadata={"help": "If True, all SLOs must be met. If False, any SLO can be met."},
    )

    @classmethod
    def from_capacity_search_config(cls, capacity_config: Any) -> "SLOSet":
        """Create SLOSet from legacy CapacitySearchConfig."""
        slos: List[BaseSLO] = []

        if capacity_config.slo_type == "deadline":
            if capacity_config.dynamic_ttft_slo:
                slos.append(
                    PredictionOffsetSLO(
                        metric=SLOMetric.TTFT,
                        value=capacity_config.ttft_slack_slo,
                        percentile=0.99,  # default percentile
                        name="Dynamic TTFT with slack",
                    )
                )
            slos.append(
                ConstantSLO(
                    metric=SLOMetric.TBT,
                    value=capacity_config.tbt_slo,
                    percentile=capacity_config.tbt_percentile,
                    name="TBT",
                )
            )
            slos.append(
                ConstantSLO(
                    metric=SLOMetric.DEADLINE_MISS_RATE,
                    value=capacity_config.deadline_miss_rate_slo,
                    percentile=capacity_config.deadline_miss_rate_percentile,
                    name="Deadline miss rate",
                )
            )
        elif capacity_config.slo_type == "tbt_ttft":
            slos.extend(
                [
                    ConstantSLO(
                        metric=SLOMetric.TBT,
                        value=capacity_config.tbt_slo,
                        percentile=capacity_config.tbt_percentile,
                        name="TBT",
                    ),
                    ConstantSLO(
                        metric=SLOMetric.TTFT,
                        value=capacity_config.ttft_slo,
                        percentile=capacity_config.ttft_percentile,
                        name="TTFT",
                    ),
                ]
            )
        elif capacity_config.slo_type == "ttft_tpot":
            slos.extend(
                [
                    ConstantSLO(
                        metric=SLOMetric.TTFT,
                        value=capacity_config.ttft_slo,
                        percentile=capacity_config.ttft_percentile,
                        name="TTFT",
                    ),
                    ConstantSLO(
                        metric=SLOMetric.TPOT,
                        value=capacity_config.tpot_slo,
                        percentile=capacity_config.tpot_percentile,
                        name="TPOT",
                    ),
                ]
            )
        return cls(slos=slos)
