from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class DeadlineReportConfig:
    """
    Config for reporting deadline metrics in benchmark results:

    - Are the deadlines met?
    - What are the miss rates for TTFT and TBT deadlines?
    - What is the smallest TBT deadline that would meet the target deadline miss rate?
    """

    ttft_deadline: float = field(
        default=0.1,
        metadata={"help": "The deadline for time to first token."},
    )
    tbt_deadline: float = field(
        default=0.05,
        metadata={"help": "The deadline for time between tokens."},
    )
    target_deadline_miss_rate: float = field(
        default=0.1,
        metadata={
            "help": "The target deadline miss rate. Used to report smallest TBT deadline that would meet it."
        },
    )
