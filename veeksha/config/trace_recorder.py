from vidhi import field, frozen_dataclass


@frozen_dataclass
class TraceRecorderConfig:
    """Configuration for request tracing"""

    enabled: bool = field(True, help="Enable recording of dispatched requests")
    include_content: bool = field(
        False,
        help="Include content of the request (channel blobs, history) in trace",
    )
