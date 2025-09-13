"""
This file contains the wrapper for the benchmarking.
"""

import os

from veeksha.benchmark import run_benchmark
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


def setup_api_environment(
    api_key=None,
    api_url=None,
):
    """Set up environment variables for OpenAI API"""
    assert api_key is not None, "API key is required"
    assert api_url is not None, "API port is required"
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_url


def run_benchmark_wrapped(
    benchmark_config: BenchmarkConfig,
):
    """Main function to run benchmark and return in-memory ServiceMetrics."""

    setup_api_environment(
        api_key=benchmark_config.api_key,
        api_url=benchmark_config.api_url,
    )

    logger.info(f"Running benchmark with config: {benchmark_config}")
    service_metrics = run_benchmark(benchmark_config)
    logger.info("Benchmark finished")
    return service_metrics
