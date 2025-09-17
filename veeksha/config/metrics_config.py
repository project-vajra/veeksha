from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.deadline_config import DeadlineReportConfig


@frozen_dataclass
class MetricsConfig:
    deadline_report: DeadlineReportConfig = field(
        default_factory=DeadlineReportConfig,
        metadata={"help": "Reporting-only deadline thresholds for derived metrics."},
    )
    should_write_metrics_to_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to write metrics to wandb."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb project to log metrics to."},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb group to log metrics to."},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb run name to log metrics to."},
    )
