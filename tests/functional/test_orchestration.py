"""
Test script for server orchestration functionality.

This script creates a minimal benchmark configuration programmatically
and tests the full server orchestration workflow:
1. Launch server
2. Health check
3. Run benchmark
4. Shutdown server

Usage:
    python test_orchestration.py
"""

import tempfile

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


def create_minimal_benchmark_config(output_dir: str) -> BenchmarkConfig:
    """Create a minimal benchmark configuration for testing."""

    # Client configuration
    client_config = ClientConfig(
        model="Qwen/Qwen3-1.7B",  # Match server model
        num_clients=1,
        num_concurrent_requests_per_client=2,  # Very small for testing
        llm_api="openai_chat",  # Use chat completions API for Qwen model
        address_append_value="chat/completions",  # Chat completions endpoint
    )

    # Metrics configuration
    metrics_config = MetricsConfig(
        output_dir=output_dir,
    )

    # Request generators (minimal config)
    interval_config = PoissonRequestIntervalGeneratorConfig(qps=0.5)  # Slow rate
    length_config = UniformRequestLengthGeneratorConfig(
        min_tokens=10,  # Very short requests
        prefill_to_decode_ratio=0.1,
    )
    request_config = SyntheticRequestGeneratorConfig(
        interval_generator_config=interval_config,
        length_generator_config=length_config,
    )

    # Benchmark configuration
    benchmark_config = BenchmarkConfig(
        client_config=client_config,
        metrics_config=metrics_config,
        request_generator_config=request_config,
        max_completed_requests=5,  # Very few requests for quick test
        timeout=60,  # Short timeout
    )

    return benchmark_config


def test_orchestration():
    """Test the full orchestration workflow."""
    logger.info("=" * 80)
    logger.info("Testing Server Orchestration")
    logger.info("=" * 80)

    # Create temporary output directory
    with tempfile.TemporaryDirectory(prefix="veeksha_test_") as output_dir:
        # Create minimal benchmark config
        logger.info("Creating minimal benchmark configuration...")
        benchmark_config = create_minimal_benchmark_config(output_dir)

        # Create server config
        logger.info("Creating server configuration...")
        server_config = ServerConfig(
            engine="vllm",
            model="Qwen/Qwen3-1.7B",  # Use Qwen3 model
            port=8000,
            tensor_parallel_size=1,
            gpu_ids=[2],  # Use GPU 2 which has more free memory
            dtype="auto",
            auto_shutdown=True,
            startup_timeout=120,  # 2 minutes should be enough for small model
        )

        # Run the orchestration
        logger.info("Starting orchestration test...")
        try:
            logger.info("Launching server...")
            with managed_server(server_config) as info:
                logger.info(f"Server ready at {info['api_base']}")
                logger.info("Running benchmark...")

                metrics = run_benchmark(benchmark_config)

                # Display results
                logger.info("=" * 80)
                logger.info("TEST PASSED - Orchestration successful!")
                logger.info("=" * 80)

                summary = metrics.get_aggregated_summary()
                for key, value in summary.items():
                    logger.info(f"{key}: {value}")

                logger.info(
                    f"\nResults saved to: {benchmark_config.metrics_config.output_dir}"
                )

            logger.info("Server shut down")
            return True

        except Exception as e:
            logger.error("=" * 80)
            logger.error("TEST FAILED - Orchestration error!")
            logger.error("=" * 80)
            logger.error(f"Error: {e}")
            raise


if __name__ == "__main__":
    success = test_orchestration()
    if success:
        logger.info("All tests passed!")
    else:
        logger.error("Tests failed!")
        exit(1)
