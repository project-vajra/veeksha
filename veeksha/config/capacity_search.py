"""
Capacity search is a meta benchmark: it runs the same benchmark configuration
multiple times while varying a single traffic-scheduler knob, and selects the
maximum value that still meets the configured SLOs.
"""

from typing import List

from vidhi import field, frozen_dataclass, parse_cli_sweep

from veeksha.cli.base import VeekshaCommand
from veeksha.config.benchmark import BenchmarkConfig


@frozen_dataclass
class CapacitySearchConfig(VeekshaCommand, name="capacity-search"):
    """Configuration for a capacity search run."""

    output_dir: str = field(
        "capacity_search_output",
        help="Output directory for capacity search artifacts and runs.",
    )
    start_value: float = field(
        1.0,
        help="Initial value to probe. The algorithm will expand upward from "
        "this value until it finds a failing point, then binary search.",
    )
    max_value: float = field(
        100.0,
        help="Ceiling for the search. Expansion will not exceed this value.",
    )
    expansion_factor: float = field(
        2.0,
        help="Factor by which to expand the search bound during probing phase. "
        "E.g., 2.0 means double the value on each passing probe.",
    )
    max_iterations: int = field(
        20,
        help="Maximum number of search iterations (probe + binary).",
    )
    precision: int = field(
        2,
        help="Decimal rounding precision for rate-based searches (float knob).",
    )
    benchmark_config: BenchmarkConfig = field(
        default_factory=BenchmarkConfig,
        help="Benchmark config used as the base for all iterations.",
    )

    @classmethod
    def create_from_cli_args(cls) -> List["CapacitySearchConfig"]:
        """Create one or more CapacitySearchConfig instances from CLI/YAML."""
        return parse_cli_sweep(cls)
