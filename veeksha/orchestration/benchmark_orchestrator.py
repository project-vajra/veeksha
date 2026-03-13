"""
Context manager for automatic server lifecycle management.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator

from veeksha.config.server import BaseServerConfig
from veeksha.orchestration.registry import ServerManagerRegistry
from veeksha.orchestration.server_manager import BaseServerManager


def create_server_manager(
    config: BaseServerConfig,
    output_dir: str,
) -> BaseServerManager:
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
            - api_base: API base URL
            - api_key: API key
            - server_manager: Server manager instance
    """
    server_manager = create_server_manager(config, output_dir=output_dir)

    try:
        launch_result = server_manager.launch()
        if isinstance(launch_result, tuple):
            success, error = launch_result
        else:
            success, error = bool(launch_result), None
        if not success:
            raise RuntimeError(f"Failed to launch server: {error}")

        if not server_manager.wait_for_ready():
            raise RuntimeError("Server failed to become ready")

        resolved_config = server_manager.config
        api_base = resolved_config.get_api_base_url()

        # Set environment variables for clients
        os.environ["OPENAI_API_KEY"] = resolved_config.api_key or "EMPTY"
        os.environ["OPENAI_API_BASE"] = api_base

        yield {
            "api_base": api_base,
            "api_key": resolved_config.api_key,
            "metrics_url": resolved_config.get_metrics_url(),
            "config": resolved_config,
            "server_manager": server_manager,
        }

    finally:
        server_manager.shutdown()
