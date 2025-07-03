import json
import os
from dataclasses import field
from typing import Optional

from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.constants.configuration_constants import DEFAULT_SEED


@frozen_dataclass
class CapacitySearchConfig:
    """Configuration for capacity search benchmark. This is a special benchmark that runs multiple benchmarks with different QPS and
    finds the maximum QPS that can be sustained given the deadline constraints."""

    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Seed for the random number generator for capacity search."},
    )
    start_qps: float = field(
        default=1,
        metadata={"help": "The starting QPS for the capacity search."},
    )
    num_qps_steps: int = field(
        default=10,
        metadata={"help": "The number of QPS steps for the capacity search."},
    )
    min_search_granularity: float = field(
        default=2.5,
        metadata={"help": "Minimum search granularity for capacity (%)"},
    )
    max_iterations: int = field(
        default=20,
        metadata={"help": "Maximum number of iterations for capacity search."},
    )
    output_dir: str = field(
        default="./veeksha/capacity_search/output",
        metadata={"help": "Output directory for capacity search."},
    )
    benchmark_config_file: str = field(
        default="./veeksha/capacity_search/config/default_config.yml",
        metadata={"help": "Path to benchmark config file."},
    )
    slo_type: str = field(
        default="deadline",
        metadata={"help": "Type of SLO to use for capacity search"},
    )
    tbt_slo: float = field(
        default=0.03,
        metadata={"help": "TBT SLO for capacity search"},
    )
    tbt_percentile: float = field(
        default=0.99,
        metadata={"help": "TBT percentile for capacity search"},
    )
    ttft_slo: float = field(
        default=0.1,
        metadata={"help": "TTFT SLO for capacity search"},
    )
    ttft_percentile: float = field(
        default=0.9,
        metadata={"help": "TTFT percentile for capacity search"},
    )
    tpot_slo: float = field(
        default=0.1,
        metadata={"help": "TPOT SLO for capacity search"},
    )
    tpot_percentile: float = field(
        default=0.9,
        metadata={"help": "TPOT percentile for capacity search"},
    )
    ttft_slack_slo: float = field(
        default=0.3,
        metadata={"help": "TTFT slack SLO for capacity search"},
    )
    deadline_miss_rate_slo: float = field(
        default=0.1,
        metadata={"help": "Deadline miss rate SLO for capacity search"},
    )
    deadline_miss_rate_percentile: float = field(
        default=0.99,
        metadata={"help": "Deadline miss rate percentile for capacity search"},
    )
    dynamic_ttft_slo: bool = field(
        default=True,
        metadata={"help": "Dynamic TTFT SLO for capacity search"},
    )
    # # TODO: remove from arg, move to trace config or similar
    # trace_session_match_threshold: float = field(
    #     default=0.9,
    #     metadata={"help": "Trace session match threshold for capacity search"},
    # )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb project for capacity search"},
    )
    enable_wandb_sweep: bool = field(
        default=False,
        metadata={"help": "Enable wandb sweep for capacity search"},
    )
    wandb_sweep_name: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb sweep name for capacity search"},
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb sweep id for capacity search"},
    )

    @classmethod
    def create_from_cli_args(cls):
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        object.__setattr__(instance, "__flat_config__", flat_config)
        return flat_config.reconstruct_original_dataclass()

    def to_dict(self):
        return self.__dict__

    def write_config_to_file(self):
        config_dict = self.to_dict()
        with open(os.path.join(f"{self.output_dir}", "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)
