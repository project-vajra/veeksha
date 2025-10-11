"""
Template for running microbenchmarks with automatic server orchestration.

This module provides a template for running parameter sweeps across different
server configurations. Adapt this template for your specific benchmarking needs.

Key Pattern:
    1. Define your server configurations (models, parallelism, batch sizes, etc.)
    2. Define your benchmark configuration
    3. Loop through server configs, launching/benchmarking/shutting down each

Example usage pattern:
    ```python
    from veeksha.orchestration.benchmark_orchestrator import run_benchmark_with_server
    from veeksha.config.server import ServerConfig
    from veeksha.config.benchmark import BenchmarkConfig

    # Load base benchmark config
    base_config = BenchmarkConfig.create_from_cli_args()[0]

    # Run across different configurations
    for tp_size in [1, 2, 4]:
        server_config = ServerConfig(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            tensor_parallel_size=tp_size,
            port=8000 + tp_size,  # Different port for each
        )

        metrics = run_benchmark_with_server(base_config, server_config)
        print(f"TP{tp_size} results:", metrics.get_aggregated_summary())
    ```
"""

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration.benchmark_orchestrator import run_benchmark_with_server

logger = init_logger(__name__)


def example_parameter_sweep():
    """Example of running a parameter sweep across tensor parallelism sizes."""

    logger.info("Starting microbenchmark parameter sweep")

    # Load benchmark configuration (you can also create it programmatically)
    benchmark_configs = BenchmarkConfig.create_from_cli_args()
    if not benchmark_configs:
        logger.error("No benchmark configuration provided via CLI!")
        return

    base_config = benchmark_configs[0]

    # Define tensor parallelism sizes to test
    tp_sizes = [1, 2, 4]

    results = []

    for tp_size in tp_sizes:
        logger.info("=" * 80)
        logger.info(f"Running with tensor_parallel_size={tp_size}")
        logger.info("=" * 80)

        # Create server configuration for this run
        server_config = ServerConfig(
            engine="vllm",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            port=8000 + tp_size,  # Use different port for each to avoid conflicts
            tensor_parallel_size=tp_size,
            gpu_ids=list(range(tp_size)),  # Use first N GPUs
            auto_shutdown=True,
        )

        try:
            # Run benchmark with this server configuration
            metrics = run_benchmark_with_server(
                benchmark_config=base_config,
                server_config=server_config,
            )

            # Collect results
            results.append(
                {
                    "tensor_parallel_size": tp_size,
                    "metrics": metrics.get_aggregated_summary(),
                }
            )

            logger.info(f"TP{tp_size} completed successfully")

        except Exception as e:
            logger.error(f"Failed for tp_size={tp_size}: {e}")
            continue

    # Print summary
    logger.info("=" * 80)
    logger.info("PARAMETER SWEEP RESULTS")
    logger.info("=" * 80)
    for result in results:
        logger.info(f"\nTP Size: {result['tensor_parallel_size']}")
        for key, value in result["metrics"].items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    example_parameter_sweep()
