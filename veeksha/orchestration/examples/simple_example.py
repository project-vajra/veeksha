"""
Example: Running a standard benchmark with automatic server orchestration.

This example demonstrates how to run a standard benchmark with
automatic server lifecycle management:
1. Launch vLLM server
2. Run benchmark
3. Automatically shutdown server

This is useful for:
- Running performance benchmarks on different models
- Testing different server configurations
- Automating benchmark workflows
"""

from veeksha.benchmark import run_benchmark
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import ClientConfig
from veeksha.config.generators.interval_generator.poisson_generator import (
    PoissonRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.length_generator.uniform_generator import (
    UniformRequestLengthGeneratorConfig,
)
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.config.metrics import MetricsConfig
from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration import managed_server

logger = init_logger(__name__)


def example_synthetic_benchmark():
    """Run a synthetic benchmark with automatic server orchestration."""

    logger.info("=" * 80)
    logger.info("Example: Synthetic Benchmark with Server Orchestration")
    logger.info("=" * 80)

    # Configure server
    server_config = ServerConfig(
        engine="vllm",
        model="Qwen/Qwen3-1.7B",
        host="localhost",
        port=8000,
        tensor_parallel_size=1,
        auto_shutdown=True,
        startup_timeout=300,
    )

    # Configure synthetic benchmark
    benchmark_config = BenchmarkConfig(
        seed=42,
        timeout=120,  # 2 minutes for demo
        max_completed_requests=10,  # Few requests for demo
        api_key=server_config.api_key,
        api_url=server_config.get_api_base_url(),
        client_config=ClientConfig(
            model="Qwen/Qwen3-1.7B",
        ),
        request_generator_config=SyntheticRequestGeneratorConfig(
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1.0),
            length_generator_config=UniformRequestLengthGeneratorConfig(
                min_tokens=10,
                prefill_to_decode_ratio=2.0,
            ),
        ),
        metrics_config=MetricsConfig(
            output_dir="./benchmark_results/synthetic_example",
        ),
    )

    # Run with orchestration
    logger.info("Launching server...")
    with managed_server(server_config) as info:
        logger.info(f"Server ready at {info['api_base']}")
        logger.info("Running benchmark...")

        metrics = run_benchmark(benchmark_config)

        logger.info("Synthetic benchmark completed!")
        logger.info(f"Results saved to: {metrics.output_dir}")
        logger.info(f"Total requests: {metrics.num_requests}")
        logger.info(f"Completed requests: {metrics.num_completed_requests}")

    logger.info("Server shut down")


if __name__ == "__main__":
    example_synthetic_benchmark()
