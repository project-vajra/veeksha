"""
Orchestration module for managing LLM inference servers and running benchmarks.

This module provides tools for:
- Launching, managing, and shutting down LLM inference servers via context manager
- Running workloads with automatic server orchestration

Example usage:
    ```python
    from veeksha.orchestration import managed_server
    from veeksha.benchmark import run_benchmark
    from veeksha.config.server import ServerConfig
    from veeksha.config.benchmark import BenchmarkConfig

    server_config = ServerConfig(
        engine="vllm",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        port=8000,
        auto_shutdown=True,
    )

    benchmark_config = BenchmarkConfig.create_from_cli_args()[0]

    with managed_server(server_config) as info:
        print(f"Server ready at {info['api_base']}")
        metrics = run_benchmark(benchmark_config)
    ```
"""

from veeksha.orchestration.benchmark_orchestrator import (
    create_server_manager,
    managed_server,
)
from veeksha.orchestration.parallel_runner import (
    ParallelBenchmarkRunner,
    SequentialJobQueue,
)
from veeksha.orchestration.resource_manager import ResourceManager
from veeksha.orchestration.server_manager import BaseServerManager
from veeksha.orchestration.vajra_server import (
    VajraServerManager,
    create_vajra_server_manager,
)
from veeksha.orchestration.vllm_server import (
    VLLMServerManager,
    create_vllm_server_manager,
)

__all__ = [
    # Server managers
    "BaseServerManager",
    "VLLMServerManager",
    "VajraServerManager",
    "create_vllm_server_manager",
    "create_vajra_server_manager",
    "create_server_manager",
    # Context manager
    "managed_server",
    # Resource management
    "ResourceManager",
    "ParallelBenchmarkRunner",
    "SequentialJobQueue",
]
