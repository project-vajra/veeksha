from dataclasses import field
from typing import Optional

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.logger import init_logger
from veeksha.types import SloType

logger = init_logger(__name__)


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
        assert 0.0 <= self.percentile <= 1.0, "percentile must be between 0.0 and 1.0"


@frozen_dataclass
class ConstantSloConfig(BaseSloConfig):
    """SLO with a fixed constant value threshold."""

    metric: str = field(
        default="ttft", metadata={"help": "The metric key this SLO applies to"}
    )

    value: float = field(
        default=-1.0,
        metadata={
            "help": "The constant value for the SLO. If a percentage, from 0 to 1. If a time, in seconds."
        },
    )

    def __post_init__(self):
        """Validate SLO definition."""
        assert self.value > 0, "ConstantSLO: value must be specified and must be > 0"

    @classmethod
    def get_type(cls) -> SloType:
        return SloType.CONSTANT
