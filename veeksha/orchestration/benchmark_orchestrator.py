"""
Context manager for automatic server lifecycle management.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator

from veeksha.config.server import BaseServerConfig
from veeksha.orchestration.managed_engines import BaseEngineRunner
from veeksha.orchestration.registry import ServerManagerRegistry


def create_server_manager(
    config: BaseServerConfig,
    output_dir: str,
) -> BaseEngineRunner:
    """Create appropriate server manager based on config type."""
    return ServerManagerRegistry.get(
        config.get_type(),
        config=config,
        output_dir=output_dir,
    )


@contextmanager
def managed_server(
    config: BaseServerConfig,
    output_dir: str,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for automatic server lifecycle management.

    Handles:
    1. Launch server
    2. Wait for ready
    3. Yield server info
    4. Shutdown

    Args:
        config: Server configuration
        output_dir: Directory for server logs.

    Yields:
        Dictionary with server info:
            - endpoint: Endpoint config
            - api_base: API base URL
            - api_key: API key
            - server_manager: Server manager instance
    """
    server_manager = create_server_manager(config, output_dir=output_dir)

    try:
        server_manager.start()
        endpoint = server_manager.get_endpoint()

        # Set environment variables for clients.
        os.environ["OPENAI_API_KEY"] = endpoint.api_key or "EMPTY"
        os.environ["OPENAI_API_BASE"] = endpoint.api_base

        yield {
            "endpoint": endpoint,
            "api_base": endpoint.api_base,
            "api_key": endpoint.api_key,
            "server_manager": server_manager,
        }

    finally:
        server_manager.stop()
