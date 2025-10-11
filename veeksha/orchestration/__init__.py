"""
Orchestration module for managing LLM inference servers and running benchmarks.

This module provides tools for:
- Launching, managing, and shutting down LLM inference servers
- Running standard benchmarks with automatic server orchestration
- Running microbenchmarks (prefill/decode probes) with server orchestration
- Running lm_eval tasks with server orchestration

Example usage:
    ```python
    from veeksha.orchestration import run_benchmark_with_server
    from veeksha.config.server import ServerConfig
    from veeksha.config.benchmark import BenchmarkConfig

    server_config = ServerConfig(
        engine="vllm",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        port=8000,
    )

    benchmark_config = BenchmarkConfig.create_from_cli_args()[0]
    metrics = run_benchmark_with_server(benchmark_config, server_config)
    ```
"""

from veeksha.orchestration.benchmark_orchestrator import (
    run_benchmark_with_server,
    run_lmeval_with_server,
    run_microbenchmark_with_server,
)
from veeksha.orchestration.server_manager import BaseServerManager
from veeksha.orchestration.vllm_server import (
    VLLMServerManager,
    create_vllm_server_manager,
)

__all__ = [
    # Server managers
    "BaseServerManager",
    "VLLMServerManager",
    "create_vllm_server_manager",
    # Orchestration functions
    "run_benchmark_with_server",
    "run_microbenchmark_with_server",
    "run_lmeval_with_server",
]
