from dataclasses import field
from typing import List

from veeksha.config.core.base_entrypoint_config import BaseEntrypointConfig
from veeksha.config.benchmark_config import BenchmarkConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.slo_config import BaseSloConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


@frozen_dataclass(allow_from_file=True)
class CapacitySearchConfig(BaseEntrypointConfig):
    """Configuration for capacity search benchmark. This is a special benchmark that runs multiple benchmarks with different QPS and
    finds the maximum QPS that can be sustained given the deadline constraints."""
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
        metadata={"help": "Minimum search granularity for capacity (%%)"},
    )
    max_iterations: int = field(
        default=20,
        metadata={"help": "Maximum number of iterations for capacity search."},
    )
    benchmark_config: BenchmarkConfig = field(
        default_factory=BenchmarkConfig,
        metadata={"help": "Benchmark config for capacity search."},
    )
    slos: List[BaseSloConfig] = field(
        default_factory=list,
        metadata={"help": "List of SLO definitions to evaluate"},
    )

    def __post_init__(self):
        assert self.start_qps >= 0, "start_qps must be greater than 0"  
        assert self.num_qps_steps >= 0, "num_qps_steps must be greater than 0"
        assert self.min_search_granularity >= 0, "min_search_granularity must be greater than 0"
        assert self.max_iterations >= 0, "max_iterations must be greater than 0"
        assert self.benchmark_config is not None, "benchmark_config must be provided"
        assert self.slos is not None, "slos must be provided"
