"""
Orchestration for running benchmarks, microbenchmarks, and lm_eval with automatic server management.

This module provides high-level interfaces for running different workload types with
automatic server lifecycle management (launch, wait for ready, run workload, shutdown).

Supported workload types:
- Standard benchmarks (via run_benchmark)
- Microbenchmarks (prefill/decode probes)
- LM-Eval tasks

All functions follow the same pattern:
1. Launch server (if server_config provided)
2. Wait for server to be ready
3. Run workload
4. Shutdown server (if auto_shutdown enabled)
"""

import os
from typing import Any, Dict, Optional

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
        raise ValueError(f"Unsupported engine: {engine}. " f"Currently supported: vllm")


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
        if (
            server_manager is not None
            and server_config is not None
            and server_config.auto_shutdown
        ):
            logger.info("=" * 80)
            logger.info("STEP 4: Shutting down server")
            logger.info("=" * 80)
            server_manager.shutdown()


def run_microbenchmark_with_server(
    microbenchmark_config: "MicrobenchmarkConfig",  # type: ignore
    server_config: Optional[ServerConfig] = None,
) -> None:
    """Run microbenchmark (prefill/decode probes) with automatic server orchestration.

    This function handles the full lifecycle:
    1. Launch server (if server_config provided)
    2. Wait for server to be ready
    3. Run microbenchmark probes
    4. Shutdown server (if auto_shutdown enabled)

    Args:
        microbenchmark_config: Microbenchmark configuration
        server_config: Server configuration (None means use existing server)
    """
    pass
    # from veeksha.microbenchmark import Microbenchmark

    # server_manager = None

    # try:
    #     # Setup server if config provided
    #     if server_config is not None:
    #         logger.info("=" * 80)
    #         logger.info("STEP 1: Launching server for microbenchmark")
    #         logger.info("=" * 80)

    #         server_manager = create_server_manager(server_config)

    #         # Launch server
    #         if not server_manager.launch():
    #             raise RuntimeError("Failed to launch server")

    #         # Wait for ready
    #         if not server_manager.wait_for_ready():
    #             raise RuntimeError("Server failed to become ready")

    #         logger.info(f"Server ready at {server_config.get_api_base_url()}")

    #         # Update microbenchmark config to use the launched server
    #         object.__setattr__(
    #             microbenchmark_config, "api_url", server_config.get_api_base_url()
    #         )
    #         object.__setattr__(microbenchmark_config, "api_key", server_config.api_key)

    #     # Run microbenchmark
    #     logger.info("=" * 80)
    #     logger.info("STEP 2: Running microbenchmark probes")
    #     logger.info("=" * 80)

    #     microbenchmark = Microbenchmark(microbenchmark_config)
    #     microbenchmark.run()

    #     logger.info("=" * 80)
    #     logger.info("STEP 3: Microbenchmark complete")
    #     logger.info("=" * 80)
    #     logger.info(f"Results saved to: {microbenchmark_config.output_dir}")

    # except Exception as e:
    #     logger.error(f"Error during microbenchmark with server orchestration: {e}")
    #     raise

    # finally:
    #     # Cleanup server
    #     if (
    #         server_manager is not None
    #         and server_config is not None
    #         and server_config.auto_shutdown
    #     ):
    #         logger.info("=" * 80)
    #         logger.info("STEP 4: Shutting down server")
    #         logger.info("=" * 80)
    #         server_manager.shutdown()


def run_lmeval_with_server(
    benchmark_config: BenchmarkConfig,
    server_config: Optional[ServerConfig] = None,
) -> Dict[str, Any]:
    """Run lm_eval tasks with automatic server orchestration.

    This function handles the full lifecycle:
    1. Launch server (if server_config provided)
    2. Wait for server to be ready
    3. Run lm_eval benchmark
    4. Shutdown server (if auto_shutdown enabled)

    Note: The benchmark_config must have request_generator_config set to
    LmevalRequestGeneratorConfig with the desired tasks.

    Args:
        benchmark_config: Benchmark configuration with lm_eval request generator
        server_config: Server configuration (None means use existing server)

    Returns:
        Dictionary containing lm_eval results
    """
    from veeksha.types import RequestGeneratorType

    server_manager = None

    try:
        # Validate config
        if (
            benchmark_config.request_generator_config.get_type()
            != RequestGeneratorType.LMEVAL
        ):
            raise ValueError(
                "benchmark_config must have request_generator_config of type LMEVAL. "
                f"Got: {benchmark_config.request_generator_config.get_type()}"
            )

        # Setup server if config provided
        if server_config is not None:
            logger.info("=" * 80)
            logger.info("STEP 1: Launching server for lm_eval")
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

        # Run lm_eval benchmark
        logger.info("=" * 80)
        logger.info("STEP 2: Running lm_eval tasks")
        logger.info("=" * 80)

        service_metrics = run_benchmark(benchmark_config)

        logger.info("=" * 80)
        logger.info("STEP 3: LM-Eval complete")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {benchmark_config.metrics_config.output_dir}")

        # Load and return lm_eval results
        import json

        results_path = os.path.join(
            benchmark_config.metrics_config.output_dir, "lmeval_results.json"
        )

        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                lmeval_results = json.load(f)
            return lmeval_results
        else:
            logger.warning(f"LM-Eval results not found at {results_path}")
            return {}

    except Exception as e:
        logger.error(f"Error during lm_eval with server orchestration: {e}")
        raise

    finally:
        # Cleanup server
        if (
            server_manager is not None
            and server_config is not None
            and server_config.auto_shutdown
        ):
            logger.info("=" * 80)
            logger.info("STEP 4: Shutting down server")
            logger.info("=" * 80)
            server_manager.shutdown()
