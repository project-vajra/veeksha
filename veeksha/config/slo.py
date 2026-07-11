from typing import Optional

from vidhi import BasePolyConfig, field, frozen_dataclass

from veeksha.types import SloType

SUPPORTED_SLO_METRICS = {
    "ttfc",
    "tbc",
    "tpot",
    "e2e",
    "generated_audio_duration",
    "rtf",
    "first_input_to_first_audio_ms",
    "post_commit_audio_delivery_ms",
    "required_startup_delay_ms",
    "zero_delay_stall_count",
    "zero_delay_total_stall_ms",
    "zero_delay_longest_stall_ms",
    "streaming_rtf",
}


@frozen_dataclass
class BaseSloConfig(BasePolyConfig):
    """Base class for a single SLO definition."""

    percentile: float = field(
        0.99, help="Percentile at which to evaluate the SLO (0.0-1.0)"
    )
    name: Optional[str] = field(None, help="Human-readable name for this SLO")

    def __post_init__(self) -> None:
        if not 0.0 <= self.percentile <= 1.0:
            raise ValueError(
                f"percentile must be between 0.0 and 1.0, got {self.percentile}"
            )


@frozen_dataclass
class ConstantSloConfig(BaseSloConfig):
    """SLO with a fixed constant value threshold."""

    metric: str = field(
        "ttfc",
        help=f"The metric key this SLO applies to. Available: {', '.join(sorted(SUPPORTED_SLO_METRICS))}.",
    )

    value: float = field(
        -1.0,
        help="The constant value for the SLO. If a percentage, from 0 to 1. If a time, in seconds.",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.metric not in SUPPORTED_SLO_METRICS:
            raise ValueError(
                f"ConstantSLO: metric '{self.metric}' is not supported. "
                f"Supported: {sorted(SUPPORTED_SLO_METRICS)}"
            )
        if self.value <= 0.0:
            raise ValueError("ConstantSLO: value must be specified and must be > 0")

    @classmethod
    def get_type(cls) -> SloType:
        return SloType.CONSTANT
