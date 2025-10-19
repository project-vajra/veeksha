"""
Example: Running benchmarks in parallel with automatic resource management.

This example demonstrates how to use ParallelBenchmarkRunner to execute
multiple benchmarks concurrently while automatically managing GPU resources.

This is useful for:
- Running parameter sweeps across multiple configurations
- Comparing performance across different models or settings
- Maximizing GPU utilization by running benchmarks in parallel
- Automated benchmarking workflows with multiple experiments
"""

import time
from typing import List, Tuple

from veeksha.benchmark import run_benchmark
from veeksha.config.benchmark import (
    BenchmarkConfig,
    ClientConfig,
    SyntheticRequestGeneratorConfig,
)
from veeksha.config.metrics import MetricsConfig
from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration import ParallelBenchmarkRunner

logger = init_logger(__name__)


def create_benchmark_configs() -> List[Tuple[ServerConfig, BenchmarkConfig]]:
    """Create a list of (server_config, benchmark_config) tuples for parallel execution.

    Returns:
        List of configuration tuples for different experiments
    """
    configs = []

    # Define experiments: (model, tensor_parallel_size, port)
    experiments = [
        ("Qwen/Qwen3-1.7B", 1, 8000),
        ("Qwen/Qwen3-1.7B", 1, 8001),
        ("meta-llama/Meta-Llama-3-8B-Instruct", 2, 8002),
    ]

    for model, tp_size, port in experiments:
        # Server configuration
        server_config = ServerConfig(
            engine="vajra",
            model=model,
            host="localhost",
            port=port,
            tensor_parallel_size=tp_size,
            auto_shutdown=True,
        )

        # Benchmark configuration
        benchmark_config = BenchmarkConfig(
            api_url=f"http://localhost:{port}/v1",
            timeout=300,
            max_completed_requests=50,  # Smaller for faster parallel execution
            client_config=ClientConfig(model=model),
            request_generator_config=SyntheticRequestGeneratorConfig(),
            metrics_config=MetricsConfig(
                output_dir=f"results/parallel_{model.split('/')[-1]}_tp{tp_size}_{int(time.time())}"
            ),
        )

        configs.append((server_config, benchmark_config))

    return configs


def example_parallel_benchmarks():
    """Run multiple benchmarks in parallel with automatic resource management."""

    logger.info("=" * 80)
    logger.info("Example: Parallel Benchmark Execution")
    logger.info("=" * 80)

    # Create benchmark configurations
    configs = create_benchmark_configs()
    logger.info(f"Created {len(configs)} benchmark configurations")

    # Initialize parallel runner with limited workers to avoid overwhelming the system
    max_concurrent = min(3, len(configs))  # Run up to 3 benchmarks concurrently
    logger.info(f"Running with max {max_concurrent} concurrent benchmarks")

    start_time = time.time()

    # Run benchmarks in parallel
    with ParallelBenchmarkRunner(max_workers=max_concurrent) as runner:
        results = runner.run(
            configs=configs,
            benchmark_func=run_benchmark,
            wait_for_resources=True,
            resource_timeout=300,  # 5 minutes timeout for resource allocation
        )

    total_time = time.time() - start_time

    # Analyze results
    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    successful = 0
    failed = 0

    for i, result in enumerate(results):
        server_config, benchmark_config = configs[i]
        model = server_config.model
        tp_size = server_config.tensor_parallel_size

        if result is not None:
            successful += 1
            logger.info(f"✅ {model} (TP={tp_size}): SUCCESS")
            logger.info(f"   Output: {result.output_dir}")
            if hasattr(result, "total_requests"):
                logger.info(f"   Requests: {result.total_requests}")
            if hasattr(result, "duration"):
                logger.info(f"   Duration: {result.duration:.2f}s")
        else:
            failed += 1
            logger.info(f"❌ {model} (TP={tp_size}): FAILED")

    logger.info("=" * 80)
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"Successful: {successful}/{len(configs)}")
    logger.info(f"Failed: {failed}/{len(configs)}")
    logger.info(".2f")
    logger.info("=" * 80)


def example_sequential_queue():
    """Example using SequentialJobQueue for fine-grained control."""

    logger.info("=" * 80)
    logger.info("Example: Sequential Job Queue")
    logger.info("=" * 80)

    from veeksha.orchestration import SequentialJobQueue

    # Create a subset of configurations for sequential execution
    configs = create_benchmark_configs()[:2]  # Just run 2 experiments

    # Initialize sequential queue
    queue = SequentialJobQueue()

    # Add jobs to queue
    for server_config, benchmark_config in configs:
        queue.add_job(server_config, benchmark_config, run_benchmark)

    logger.info(f"Added {len(configs)} jobs to sequential queue")

    # Execute all jobs sequentially
    start_time = time.time()
    results = queue.execute_all(wait_for_resources=True)
    total_time = time.time() - start_time

    # Report results
    logger.info("=" * 80)
    logger.info("SEQUENTIAL QUEUE RESULTS")
    logger.info("=" * 80)

    successful = sum(1 for r in results if r is not None)
    failed = sum(1 for r in results if r is None)

    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"Successful: {successful}/{len(results)}")
    logger.info(f"Failed: {failed}/{len(results)}")
    logger.info("=" * 80)


def main():
    """Run the parallel benchmark examples."""

    logger.info("Starting Parallel Runner Examples")
    logger.info("=" * 80)

    try:
        # Example 1: Parallel execution
        example_parallel_benchmarks()

        # Brief pause between examples
        time.sleep(5)

        # Example 2: Sequential queue
        example_sequential_queue()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        return 1

    logger.info("All examples completed!")
    return 0


if __name__ == "__main__":
    exit(main())
