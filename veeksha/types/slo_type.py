from enum import Enum


class SloType(Enum):
    """Enum for different SLO types."""

    CONSTANT = "constant"
    TTFT_PREDICTION_MULTIPLIER = "ttft_prediction_multiplier"
    TTFT_PREDICTION_OFFSET = "ttft_prediction_offset"
    DEADLINE = "deadline"
