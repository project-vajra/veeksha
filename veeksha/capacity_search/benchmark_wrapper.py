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
    """Main function to run benchmark and return in-memory ServiceMetrics."""

    logger.info(f"Running benchmark with config: {benchmark_config}")

    # Initialize dashboard if enabled
    if benchmark_config.dashboard_config.enabled:
        from veeksha.dashboard.handler import init_dashboard_event_processor, stop_dashboard_event_processor
        init_dashboard_event_processor(
            enabled=True,
            enable_frontend=False,
            max_queue_size=benchmark_config.dashboard_config.max_queue_size,
            max_live_requests=benchmark_config.dashboard_config.max_live_requests,
        )

    try:
        service_metrics = run_benchmark(benchmark_config)
        logger.info("Benchmark finished")
        return service_metrics
    finally:
        if benchmark_config.dashboard_config.enabled:
            stop_dashboard_event_processor()
