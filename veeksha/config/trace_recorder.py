from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class TraceRecorderConfig:
    """Configuration for request tracing"""

    enabled: bool = field(
        default=True, metadata={"help": "Enable recording of dispatched requests"}
    )
    include_content: bool = field(
        default=False,
        metadata={
            "help": "Include content of the request (channel blobs, history) in trace"
        },
    )
