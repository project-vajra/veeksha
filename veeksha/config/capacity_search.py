import json
import os
from dataclasses import field
from typing import List, Optional

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.dashboard import DashboardConfig
from veeksha.config.slo import BaseSloConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


@frozen_dataclass(allow_from_file=True)
class CapacitySearchConfig:
    """Configuration for capacity search benchmark.

    This special benchmark runs multiple benchmarks at different QPS values or
    buffer sizes to find the maximum value under the specified SLOs.
    """

    search_mode: str = field(
        default="qps",
        metadata={
            "help": "Search mode: 'qps' to search for maximum QPS, or 'buffer_size' to search for maximum concurrent sessions buffer size."
        },
    )
    dashboard_config: Optional[DashboardConfig] = field(
        default=None,
        metadata={
            "help": "Dashboard configuration for capacity search. If not specified, falls back to benchmark_config.dashboard_config."
        },
    )
    start_qps: float = field(
        default=1,
        metadata={"help": "The starting QPS for the capacity search (or starting buffer size if search_mode='buffer_size')."},
    )
    num_qps_steps: int = field(
        default=10,
        metadata={"help": "The number of QPS steps for the capacity search."},
    )
    min_search_granularity: float = field(
        default=2.5,
        metadata={"help": "Minimum search granularity for capacity (%%)"},
    )
    max_iterations: int = field(
        default=20,
        metadata={"help": "Maximum number of iterations for capacity search."},
    )
    output_dir: str = field(
        default="capsearch_results",
        metadata={
            "help": "Output directory for capacity search benchmarks and results. Overrides output_dir in benchmark_config.metrics_config."
        },
    )
    benchmark_config: BenchmarkConfig = field(
        default_factory=BenchmarkConfig,
        metadata={"help": "Benchmark config for capacity search."},
    )
    slos: List[BaseSloConfig] = field(
        default_factory=list,
        metadata={"help": "List of SLO definitions to evaluate"},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={
            "help": "Wandb project for capacity search; overrides nested metrics_config.wandb_project and enables wandb logging for per-QPS attempts and the summary run."
        },
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional Wandb group; overrides nested metrics_config.wandb_group for all per-QPS attempts and the summary run."
        },
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
        """Create CapacitySearchConfig instances from CLI

        Returns:
            List of CapacitySearchConfig instances (single or
            multiple configs if YAML expands to multiple configurations)
        """
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        instances = []
        for flat_config in flat_configs:
            instance = flat_config.reconstruct_original_dataclass()
            object.__setattr__(instance, "__flat_config__", flat_config)
            instances.append(instance)
        return instances

    def get_dashboard_config(self) -> DashboardConfig:
        """Get the effective dashboard config.

        Returns the top-level dashboard_config if specified, otherwise falls back
        to benchmark_config.dashboard_config.
        """
        if self.dashboard_config is not None:
            return self.dashboard_config
        return self.benchmark_config.dashboard_config

    def to_dict(self):
        return self.__dict__

    def write_config_to_file(self):
        config_dict = self.to_dict()
        output_path = os.path.join(self.output_dir, "config.json")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=4)
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write config to {output_path}: {e}")
