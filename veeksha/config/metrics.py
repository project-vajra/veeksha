from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.deadline import DeadlineReportConfig


@frozen_dataclass(allow_from_file=True)
class MetricsConfig:
    output_dir: str = field(
        default="benchmark_results",
        metadata={"help": "The directory to save the benchmark results to."},
    )
    should_use_given_dir: bool = field(
        default=True,
        metadata={
            "help": "Whether to add directly use output_dir directory or create new directories for the results."
        },
    )
    deadline_reporting: DeadlineReportConfig = field(
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
    enable_wandb_sweep: bool = field(
        default=False,
        metadata={"help": "Whether to enable wandb sweep."},
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb sweep id to log metrics to."},
    )
    wandb_sweep_name: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb sweep name to log metrics to."},
    )
