"""
Capacity search is a meta benchmark: it runs the same benchmark configuration
multiple times while varying a single traffic-scheduler knob, and selects the
maximum value that still meets the configured SLOs.
"""

from dataclasses import field
from typing import List

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class CapacitySearchConfig:
    """Configuration for a capacity search run."""

    output_dir: str = field(
        default="capacity_search_output",
        metadata={"help": "Output directory for capacity search artifacts and runs."},
    )
    start_value: float = field(
        default=1.0,
        metadata={
            "help": "Initial value to probe. The algorithm will expand upward from "
            "this value until it finds a failing point, then binary search."
        },
    )
    max_value: float = field(
        default=100.0,
        metadata={
            "help": "Ceiling for the search. Expansion will not exceed this value."
        },
    )
    expansion_factor: float = field(
        default=2.0,
        metadata={
            "help": "Factor by which to expand the search bound during probing phase. "
            "E.g., 2.0 means double the value on each passing probe."
        },
    )
    max_iterations: int = field(
        default=20,
        metadata={"help": "Maximum number of search iterations (probe + binary)."},
    )
    precision: int = field(
        default=2,
        metadata={
            "help": "Decimal rounding precision for rate-based searches (float knob)."
        },
    )
    benchmark_config: BenchmarkConfig = field(
        default_factory=BenchmarkConfig,
        metadata={"help": "Benchmark config used as the base for all iterations."},
    )

    @classmethod
    def create_from_cli_args(cls) -> List["CapacitySearchConfig"]:
        """Create one or more CapacitySearchConfig instances from CLI/YAML."""
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        instances: List[CapacitySearchConfig] = []
        for flat_config in flat_configs:
            instance = flat_config.reconstruct_original_dataclass()
            object.__setattr__(instance, "__flat_config__", flat_config)
            instances.append(instance)
        return instances
