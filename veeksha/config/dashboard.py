from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class DashboardConfig:
    enabled: bool = field(default=False, metadata={"help": "Enable real-time dashboard"})
    max_live_requests: int = field(
        default=50, metadata={"help": "Maximum number of live requests to track"}
    )
    max_queue_size: int = field(
        default=1000, metadata={"help": "Maximum dashboard event queue size"}
    )
    chart_window_seconds: Optional[float] = field(
        default=None,
        metadata={"help": "Time window in seconds for chart x-axis. If None, shows full history."}
    )
