"""
This file contains the wrapper for the benchmarking.
"""

from veeksha.benchmark import run_benchmark
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


def run_benchmark_wrapped(
    benchmark_config: BenchmarkConfig,
):
    """Main function to run benchmark and return in-memory ServiceMetrics.

    Note: Dashboard initialization is handled by the search manager, not here.
    """

    logger.info(f"Running benchmark with config: {benchmark_config}")
    service_metrics = run_benchmark(benchmark_config)
    logger.info("Benchmark finished")
    return service_metrics
