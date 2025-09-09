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
        metadata={"help": "The deadline, in seconds, for time to first token."},
    )
    tbt_deadline: float = field(
        default=0.05,
        metadata={"help": "The deadline, in seconds, for time between tokens."},
    )
    target_deadline_miss_rate: float = field(
        default=0.1,
        metadata={
            "help": "The target deadline miss rate [0,1]. Used to report smallest TBT deadline that would meet it.",
        },
    )
    ttft_slack: float = field(
        default=0.0,
        metadata={"help": "The slack, in seconds, to add to TTFT predictions when using prefill predictions."},
    )

    def __post_init__(self):
        if self.ttft_deadline <= 0:
            raise ValueError("ttft_deadline must be greater than 0")
        if self.tbt_deadline <= 0:
            raise ValueError("tbt_deadline must be greater than 0")
        if not (0.0 <= self.target_deadline_miss_rate <= 1.0):
            raise ValueError("target_deadline_miss_rate must be in [0, 1]")
