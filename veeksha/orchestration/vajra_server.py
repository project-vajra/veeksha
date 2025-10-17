"""
Vajra server manager for orchestrating Vajra inference servers.

This module provides a concrete implementation of BaseServerManager for Vajra,
handling Vajra-specific launch commands and configuration.
"""

from typing import List, Optional

from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration.server_manager import BaseServerManager

logger = init_logger(__name__)


class VajraServerManager(BaseServerManager):
    """Manager for Vajra inference servers.

    This class handles launching and managing Vajra servers with
    proper command-line arguments and configuration.
    """

    def __init__(self, config: ServerConfig):
        super().__init__(config)

        if config.engine.lower() != "vajra":
            logger.warning(
                f"VajraServerManager created with engine='{config.engine}'. Expected 'vajra'"
            )

    def _build_launch_command(self) -> List[str]:
        """Build the Vajra server launch command.

        Returns:
            List of command arguments
        """
        import sys

        command = [
            sys.executable,
            "-m",
            "vajra.entrypoints.openai.server",
            "--model_config_model",
            self.config.model,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
        ]

        # API key is optional in Vajra's server; only pass if provided
        if self.config.api_key and self.config.api_key.lower() != "null":
            command.extend(["--api_key", self.config.api_key])

        # Tensor parallelism
        if self.config.tensor_parallel_size and self.config.tensor_parallel_size > 1:
            command.extend(["--parallel_config_tensor_parallel_size", str(self.config.tensor_parallel_size)])

        # max model length / context
        if self.config.max_model_len is not None:
            command.extend(["--model_config_max_model_len", str(self.config.max_model_len)])

        # Process additional arguments (parsed from JSON string in __post_init__)
        for key, value in self.config.additional_args_dict.items():
            # Normalize key names: replace underscores with dashes for CLI
            cli_key = key.replace("_", "-")
            if value is True:
                # Boolean flags
                command.append(f"--{cli_key}")
            elif value is False or value is None:
                # Skip false/none values
                continue
            elif isinstance(value, (list, tuple)):
                # List values => pass as repeated args
                for v in value:
                    command.extend([f"--{cli_key}", str(v)])
            else:
                # Regular key-value pairs
                command.extend([f"--{cli_key}", str(value)])

        # Allow extra Vajra-specific args that need special handling
        command.extend(self._parse_additional_vajra_args())

        return command

    def _parse_additional_vajra_args(self) -> List[str]:
        """Parse additional Vajra-specific arguments that need special handling.

        Returns:
            List of formatted command-line arguments
        """
        args: List[str] = []

        # Handle complex resource mapping configuration
        if "inference_engine_config_global_resource_mapping" in self.config.additional_args_dict:
            import json

            mapping = self.config.additional_args_dict["inference_engine_config_global_resource_mapping"]
            args.extend(["--inference_engine_config_global_resource_mapping", json.dumps(mapping)])

        return args

    def health_check(self) -> bool:
        """Check if Vajra server is healthy.

        Vajra does not expose a `/health` endpoint; use the OpenAI-compatible
        `/v1/models` endpoint as a readiness probe. If an API key is configured,
        include it in the Authorization header.
        """
        try:
            import requests

            url = f"http://{self.config.host}:{self.config.port}/v1/models"
            headers = {}
            if self.config.api_key and self.config.api_key.lower() != "null":
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            response = requests.get(url, headers=headers, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Vajra health check failed: {e}")
            return False


def create_vajra_server_manager(
    model: str,
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_ids: Optional[List[int]] = None,
    **kwargs,
) -> VajraServerManager:
    """Convenience function to create a Vajra server manager.

    Args:
        model: Model name or path
        port: Server port
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_ids: List of specific GPU IDs to use
        **kwargs: Additional configuration parameters

    Returns:
        Configured VajraServerManager instance
    """
    config = ServerConfig(
        engine="vajra",
        model=model,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        gpu_ids=gpu_ids,
        **kwargs,
    )

    return VajraServerManager(config)
