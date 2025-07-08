from veeksha.types.base_int_enum import BaseIntEnum


class SLOType(BaseIntEnum):
    """Enum for different SLO types."""
    
    CONSTANT = 1
    TTFT_PREDICTION_MULTIPLIER = 2
    TTFT_PREDICTION_OFFSET = 3
    DEADLINE = 4