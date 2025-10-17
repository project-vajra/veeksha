"""
Parallel runner for executing multiple benchmarks with resource management.

This module provides utilities for running multiple benchmarks in parallel
while efficiently managing GPU resources.
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration.benchmark_orchestrator import managed_server
from veeksha.orchestration.resource_manager import ResourceManager, ResourceMapping

logger = init_logger(__name__)


class ParallelBenchmarkRunner:
    """Runner for executing benchmarks in parallel with resource management.

    This class orchestrates the execution of multiple benchmarks, automatically
    managing server lifecycles and GPU allocation.

    Example:
        >>> runner = ParallelBenchmarkRunner(max_workers=4)
        >>> configs = [(server_cfg1, bench_cfg1), (server_cfg2, bench_cfg2)]
        >>> results = runner.run(configs, benchmark_func=run_benchmark)
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        resource_manager: Optional[ResourceManager] = None,
    ):
        """Initialize the parallel runner.

        Args:
            max_workers: Maximum number of concurrent workers (None = unlimited)
            resource_manager: Resource manager instance (creates one if None)
        """
        self.max_workers = max_workers
        self.resource_manager = resource_manager or ResourceManager()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: List[Future] = []

    def run(
        self,
        configs: List[Tuple[ServerConfig, BenchmarkConfig]],
        benchmark_func: Callable[[BenchmarkConfig], Any],
        wait_for_resources: bool = True,
        resource_timeout: Optional[float] = None,
    ) -> List[Any]:
        """Run benchmarks in parallel with automatic resource management.

        Args:
            configs: List of (ServerConfig, BenchmarkConfig) tuples
            benchmark_func: Function to run benchmarks (e.g., run_benchmark)
            wait_for_resources: If True, wait for resources to become available
            resource_timeout: Timeout for resource allocation (None = wait indefinitely)

        Returns:
            List of benchmark results in the same order as configs
        """
        logger.info(f"Starting parallel execution of {len(configs)} benchmarks")

        # Sort configs by GPU requirement (largest first to reduce fragmentation)
        sorted_configs = sorted(
            enumerate(configs),
            key=lambda x: x[1][0].tensor_parallel_size,
            reverse=True,
        )

        futures_with_indices: List[Tuple[int, Future]] = []

        for original_idx, (server_config, benchmark_config) in sorted_configs:
            # Submit task
            future = self.executor.submit(
                self._run_single_benchmark,
                server_config,
                benchmark_config,
                benchmark_func,
                wait_for_resources,
                resource_timeout,
            )
            futures_with_indices.append((original_idx, future))

        # Wait for all tasks and collect results
        results_dict: Dict[int, Any] = {}
        for original_idx, future in futures_with_indices:
            try:
                result = future.result()
                results_dict[original_idx] = result
            except Exception as e:
                logger.error(
                    f"Benchmark at index {original_idx} failed with error: {e}",
                    exc_info=True,
                )
                results_dict[original_idx] = None

        # Return results in original order
        results = [results_dict[i] for i in range(len(configs))]

        logger.info(f"Completed {len(configs)} benchmarks")
        return results

    def _run_single_benchmark(
        self,
        server_config: ServerConfig,
        benchmark_config: BenchmarkConfig,
        benchmark_func: Callable[[BenchmarkConfig], Any],
        wait_for_resources: bool,
        resource_timeout: Optional[float],
    ) -> Any:
        """Run a single benchmark with resource management.

        Args:
            server_config: Server configuration
            benchmark_config: Benchmark configuration
            benchmark_func: Function to run the benchmark
            wait_for_resources: Whether to wait for resources
            resource_timeout: Timeout for resource allocation

        Returns:
            Benchmark result
        """
        job_id = f"{server_config.model}_{server_config.tensor_parallel_size}_{int(time.time() * 1000)}"
        num_gpus = server_config.tensor_parallel_size
        resource_mapping: Optional[ResourceMapping] = None

        try:
            # Allocate resources
            if wait_for_resources:
                logger.info(f"Waiting for {num_gpus} GPUs for {job_id}")
                resource_mapping = self.resource_manager.wait_for_resources(
                    num_gpus=num_gpus, timeout=resource_timeout, job_id=job_id
                )
            else:
                resource_mapping = self.resource_manager.allocate_resources(
                    num_gpus=num_gpus, job_id=job_id
                )

            if resource_mapping is None:
                logger.error(f"Failed to allocate resources for {job_id}")
                return None

            # Extract GPU IDs from resource mapping (single-node case)
            # Assuming all GPUs are on the same node for now
            gpu_ids = [gpu_id for _, gpu_id in resource_mapping]

            # Update server config with allocated GPUs
            server_config_with_gpus = ServerConfig(
                engine=server_config.engine,
                model=server_config.model,
                host=server_config.host,
                port=server_config.port,
                api_key=server_config.api_key,
                tensor_parallel_size=server_config.tensor_parallel_size,
                gpu_ids=gpu_ids,
                dtype=server_config.dtype,
                max_model_len=server_config.max_model_len,
                additional_args=server_config.additional_args,
                startup_timeout=server_config.startup_timeout,
                health_check_interval=server_config.health_check_interval,
                auto_shutdown=server_config.auto_shutdown,
            )

            logger.info(
                f"Starting benchmark {job_id} with GPUs {gpu_ids} on port {server_config.port}"
            )

            # Use managed_server context manager for proper lifecycle management
            with managed_server(server_config_with_gpus) as server_info:
                logger.info(f"Server ready at {server_info['api_base']}")
                
                # Run benchmark
                result = benchmark_func(benchmark_config)

            logger.info(f"Completed benchmark {job_id}")
            return result

        except Exception as e:
            logger.error(f"Error running benchmark {job_id}: {e}", exc_info=True)
            return None

        finally:
            # Always release resources
            if resource_mapping:
                self.resource_manager.release_resources(job_id)

    def shutdown(self):
        """Shutdown the executor and cleanup resources."""
        logger.info("Shutting down parallel runner")
        self.executor.shutdown(wait=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


class SequentialJobQueue:
    """Queue for running jobs sequentially with resource management.

    This is useful when you want fine-grained control over job execution
    or when jobs have dependencies.

    Example:
        >>> queue = SequentialJobQueue()
        >>> for config in configs:
        >>>     queue.add_job(server_config, benchmark_config, run_benchmark)
        >>> results = queue.execute_all()
    """

    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        """Initialize the job queue.

        Args:
            resource_manager: Resource manager instance (creates one if None)
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.jobs: List[
            Tuple[ServerConfig, BenchmarkConfig, Callable[[BenchmarkConfig], Any]]
        ] = []
        self.results: List[Any] = []

    def add_job(
        self,
        server_config: ServerConfig,
        benchmark_config: BenchmarkConfig,
        benchmark_func: Callable[[BenchmarkConfig], Any],
    ):
        """Add a job to the queue.

        Args:
            server_config: Server configuration
            benchmark_config: Benchmark configuration
            benchmark_func: Function to run the benchmark
        """
        self.jobs.append((server_config, benchmark_config, benchmark_func))

    def execute_all(self, wait_for_resources: bool = True) -> List[Any]:
        """Execute all jobs in the queue sequentially.

        Args:
            wait_for_resources: If True, wait for resources to become available

        Returns:
            List of results from all jobs
        """
        logger.info(f"Executing {len(self.jobs)} jobs sequentially")

        for idx, (server_config, benchmark_config, benchmark_func) in enumerate(
            self.jobs
        ):
            logger.info(
                f"Starting job {idx + 1}/{len(self.jobs)}: "
                f"{server_config.model} (TP={server_config.tensor_parallel_size})"
            )

            job_id = f"job_{idx}_{int(time.time() * 1000)}"
            num_gpus = server_config.tensor_parallel_size
            resource_mapping: Optional[ResourceMapping] = None

            try:
                # Allocate resources
                if wait_for_resources:
                    resource_mapping = self.resource_manager.wait_for_resources(
                        num_gpus=num_gpus, job_id=job_id
                    )
                else:
                    resource_mapping = self.resource_manager.allocate_resources(
                        num_gpus=num_gpus, job_id=job_id
                    )

                if resource_mapping is None:
                    logger.error(f"Failed to allocate resources for job {idx}")
                    self.results.append(None)
                    continue

                # Extract GPU IDs
                gpu_ids = [gpu_id for _, gpu_id in resource_mapping]

                # Update server config with allocated GPUs
                server_config_with_gpus = ServerConfig(
                    engine=server_config.engine,
                    model=server_config.model,
                    host=server_config.host,
                    port=server_config.port,
                    api_key=server_config.api_key,
                    tensor_parallel_size=server_config.tensor_parallel_size,
                    gpu_ids=gpu_ids,
                    dtype=server_config.dtype,
                    max_model_len=server_config.max_model_len,
                    additional_args=server_config.additional_args,
                    startup_timeout=server_config.startup_timeout,
                    health_check_interval=server_config.health_check_interval,
                    auto_shutdown=server_config.auto_shutdown,
                )

                # Use managed_server context manager
                with managed_server(server_config_with_gpus) as server_info:
                    logger.info(f"Server ready at {server_info['api_base']}")
                    
                    # Run benchmark
                    result = benchmark_func(benchmark_config)
                    self.results.append(result)

                logger.info(f"Completed job {idx + 1}/{len(self.jobs)}")

            except Exception as e:
                logger.error(f"Error running job {idx}: {e}", exc_info=True)
                self.results.append(None)

            finally:
                # Always release resources
                if resource_mapping:
                    self.resource_manager.release_resources(job_id)

        logger.info(f"Completed all {len(self.jobs)} jobs")
        return self.results
