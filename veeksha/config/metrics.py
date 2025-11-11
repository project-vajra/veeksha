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
    dump_dispatched_requests: bool = field(
        default=True,
        metadata={
            "help": "Whether to dump dispatched requests to a JSONL file in streaming fashion."
        },
    )
    dispatched_requests_file: str = field(
        default="dispatched_requests.jsonl",
        metadata={
            "help": "Filename for dumped dispatched requests (saved in output_dir)."
        },
    )
    dump_request_metrics: bool = field(
        default=True,
        metadata={
            "help": "Whether to dump request metrics to a JSONL file in streaming fashion."
        },
    )
    request_metrics_file: str = field(
        default="request_metrics.jsonl",
        metadata={
            "help": "Filename for dumped request metrics (saved in output_dir)."
        },
    )
    dump_input_output: bool = field(
        default=True,
        metadata={
            "help": "Whether to dump input prompts and generated outputs to a JSONL file in streaming fashion."
        },
    )
    input_output_file: str = field(
        default="input_output.jsonl",
        metadata={
            "help": "Filename for dumped input/output data (saved in output_dir)."
        },
    )
