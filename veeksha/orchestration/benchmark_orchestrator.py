"""
Wrapper for running benchmarks with automatic server orchestration.

This module provides a high-level interface for running benchmarks with
automatic server lifecycle management (launch, benchmark, shutdown).
"""

import os
from typing import Optional

from veeksha.benchmark import run_benchmark
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics
from veeksha.orchestration.server_manager import BaseServerManager
from veeksha.orchestration.vllm_server import VLLMServerManager

logger = init_logger(__name__)


def create_server_manager(config: ServerConfig) -> BaseServerManager:
    """Create appropriate server manager based on config.
    
    Args:
        config: Server configuration
        
    Returns:
        Server manager instance
        
    Raises:
        ValueError: If engine is not supported
    """
    engine = config.engine.lower()
    
    if engine == "vllm":
        return VLLMServerManager(config)
    else:
        raise ValueError(
            f"Unsupported engine: {engine}. "
            f"Currently supported: vllm"
        )


def run_benchmark_with_server(
    benchmark_config: BenchmarkConfig,
    server_config: Optional[ServerConfig] = None,
) -> ServiceMetrics:
    """Run benchmark with automatic server orchestration.
    
    This function handles the full lifecycle:
    1. Launch server (if server_config provided)
    2. Wait for server to be ready
    3. Run benchmark
    4. Shutdown server (if auto_shutdown enabled)
    
    Args:
        benchmark_config: Benchmark configuration
        server_config: Server configuration (None means use existing server)
        
    Returns:
        ServiceMetrics containing benchmark results
    """
    server_manager = None
    
    try:
        # Setup server if config provided
        if server_config is not None:
            logger.info("=" * 80)
            logger.info("STEP 1: Launching server")
            logger.info("=" * 80)
            
            server_manager = create_server_manager(server_config)
            
            # Launch server
            if not server_manager.launch():
                raise RuntimeError("Failed to launch server")
            
            # Wait for ready
            if not server_manager.wait_for_ready():
                raise RuntimeError("Server failed to become ready")
            
            # Set environment variables for benchmark
            os.environ["OPENAI_API_KEY"] = server_config.api_key
            os.environ["OPENAI_API_BASE"] = server_config.get_api_base_url()
            
            logger.info(f"Server ready at {server_config.get_api_base_url()}")
        
        # Run benchmark
        logger.info("=" * 80)
        logger.info(f"STEP 2: Running benchmark")
        logger.info("=" * 80)
        
        service_metrics = run_benchmark(benchmark_config)
        
        logger.info("=" * 80)
        logger.info("STEP 3: Benchmark complete")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {benchmark_config.metrics_config.output_dir}")
        
        return service_metrics
        
    except Exception as e:
        logger.error(f"Error during benchmark with server orchestration: {e}")
        raise
        
    finally:
        # Cleanup server
        if server_manager is not None and server_config is not None and server_config.auto_shutdown:
            logger.info("=" * 80)
            logger.info("STEP 4: Shutting down server")
            logger.info("=" * 80)
            server_manager.shutdown()


def run_microbenchmark_sweep(
    base_benchmark_config: BenchmarkConfig,
    base_server_config: ServerConfig,
    parameter_sweep: dict,
) -> list[ServiceMetrics]:
    """Run a sweep of microbenchmarks with different configurations.
    
    This is useful for running multiple benchmarks across different
    server configurations (e.g., different batch sizes, models, etc.)
    while efficiently managing resources.
    
    Args:
        base_benchmark_config: Base benchmark configuration
        base_server_config: Base server configuration
        parameter_sweep: Dict mapping parameter names to lists of values
                        Example: {"tensor_parallel_size": [1, 2, 4],
                                 "max_num_seqs": [256, 512]}
    
    Returns:
        List of ServiceMetrics for each configuration
    """
    # This is a simplified version - a full implementation would:
    # 1. Generate all parameter combinations
    # 2. For each combination, create config, launch server, run benchmark, shutdown
    # 3. Collect and return all results
    
    logger.info("=" * 80)
    logger.info("MICROBENCHMARK SWEEP")
    logger.info("=" * 80)
    
    results = []
    
    # For MVP, just document the pattern
    logger.warning(
        "Microbenchmark sweep is a template. "
        "See run_microbenchmark.py for usage examples."
    )
    
    return results
