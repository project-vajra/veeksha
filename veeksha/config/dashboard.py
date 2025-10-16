from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class DashboardConfig:
    enabled: bool = field(
        default=False, metadata={"help": "Enable real-time dashboard"}
    )
    max_live_requests: int = field(
        default=50,
        metadata={"help": "Maximum number of live requests to track concurrently"},
    )
    max_queue_size: int = field(
        default=1000, metadata={"help": "Maximum number of dashboard events to queue"}
    )
    chart_window_seconds: Optional[float] = field(
        default=None,
        metadata={
            "help": "Time window in seconds for chart x-axis. If None, shows full history."
        },
    )

    def __post_init__(self):
        if self.enabled:
            if self.max_live_requests <= 0:
                raise ValueError("max_live_requests must be greater than 0")
            if self.max_queue_size <= 0:
                raise ValueError("max_queue_size must be greater than 0")
            if self.chart_window_seconds is not None and self.chart_window_seconds <= 0:
                raise ValueError("chart_window_seconds must be greater than 0")
