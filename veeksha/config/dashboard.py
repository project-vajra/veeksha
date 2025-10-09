from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class DashboardConfig:
    enabled: bool = field(default=True, metadata={"help": "Enable real-time dashboard"})
    max_live_requests: int = field(
        default=50, metadata={"help": "Maximum number of live requests to track"}
    )
    max_queue_size: int = field(
        default=1000, metadata={"help": "Maximum dashboard event queue size"}
    )
