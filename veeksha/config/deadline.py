
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from dataclasses import field

@frozen_dataclass
class DeadlineConfig:
    ttft_deadline: float = field(
        default=0.1,
        metadata={"help": "The deadline for time to first token."},
    )
    tbt_deadline: float = field(
        default=0.05,
        metadata={"help": "The deadline between tokens."},
    )
    target_deadline_miss_rate: float = field(
        default=0.1,
        metadata={"help": "The target deadline miss rate."},
    )
    ttft_slack: float = field(
        default=0.0,
        metadata={
            "help": "The slack for time to first token. Only used if use_predictions_for_ttft is True."
        },
    )