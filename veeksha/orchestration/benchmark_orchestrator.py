"""
Context manager for automatic server lifecycle management.

This module provides a clean context manager interface for running workloads
with automatic server management. Users control their own logging.

Example:
    >>> with managed_server(server_config) as server_info:
    >>>     # Server is launched and ready
    >>>     print(f"Server at {server_info['api_base']}")
    >>>     # Run your workload
    >>>     results = run_benchmark(benchmark_config)
    >>> # Server automatically shut down (if auto_shutdown=True)
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator

from veeksha.config.server import ServerConfig
from veeksha.orchestration.server_manager import BaseServerManager
from veeksha.orchestration.sglang_server import SGLangServerManager
from veeksha.orchestration.vajra_server import VajraServerManager
from veeksha.orchestration.vllm_server import VLLMServerManager


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
    elif engine == "vajra":
        return VajraServerManager(config)
    elif engine == "sglang":
        return SGLangServerManager(config)
    else:
        raise ValueError(
            f"Unsupported engine: {engine}. Currently supported: vllm, vajra, sglang"
        )


@contextmanager
def managed_server(
    config: ServerConfig,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for automatic server lifecycle management.

    Handles:
    1. Launch server
    2. Wait for ready
    3. Set environment variables
    4. Yield server info
    5. Shutdown (if auto_shutdown=True)

    Args:
        config: Server configuration

    Yields:
        Dictionary with server info:
            - api_base: API base URL
            - api_key: API key
            - server_manager: Server manager instance

    Raises:
        RuntimeError: If server fails to launch or become ready

    Example:
        >>> with managed_server(server_config) as info:
        >>>     print(f"Server ready at {info['api_base']}")
        >>>     # Run workload using info['api_base'] and info['api_key']
    """
    server_manager = create_server_manager(config)

    try:
        # Launch server
        if not server_manager.launch():
            raise RuntimeError("Failed to launch server")

        # Wait for ready
        if not server_manager.wait_for_ready():
            raise RuntimeError("Server failed to become ready")

        # Set environment variables
        api_base = config.get_api_base_url()
        os.environ["OPENAI_API_KEY"] = config.api_key
        os.environ["OPENAI_API_BASE"] = api_base

        # Yield server info
        yield {
            "api_base": api_base,
            "api_key": config.api_key,
            "server_manager": server_manager,
        }

    finally:
        # Cleanup if auto_shutdown enabled
        if config.auto_shutdown:
            server_manager.shutdown()
