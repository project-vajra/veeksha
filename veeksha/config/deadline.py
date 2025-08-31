from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
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
