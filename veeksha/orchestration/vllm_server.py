"""
vLLM server manager for orchestrating vLLM inference servers.

This module provides a concrete implementation of ServerManager for vLLM,
handling vLLM-specific launch commands and configuration.
"""

from typing import List, Optional

from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration.server_manager import BaseServerManager

logger = init_logger(__name__)


class VLLMServerManager(BaseServerManager):
    """Manager for vLLM inference servers.

    This class handles launching and managing vLLM servers with
    proper command-line arguments and configuration.
    """

    def __init__(self, config: ServerConfig):
        """Initialize vLLM server manager.

        Args:
            config: Server configuration
        """
        super().__init__(config)

        if config.engine.lower() != "vllm":
            logger.warning(
                f"VLLMServerManager created with engine='{config.engine}'. "
                "Expected 'vllm'"
            )

    def _build_launch_command(self) -> List[str]:
        """Build the vLLM server launch command.

        Returns:
            List of command arguments
        """
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--api-key",
            self.config.api_key,
        ]

        # Add tensor parallelism
        if self.config.tensor_parallel_size > 1:
            command.extend(
                ["--tensor-parallel-size", str(self.config.tensor_parallel_size)]
            )

        # Add dtype
        if self.config.dtype:
            command.extend(["--dtype", self.config.dtype])

        # Add max model length if specified
        if self.config.max_model_len is not None:
            command.extend(["--max-model-len", str(self.config.max_model_len)])

        # Process additional arguments (parsed from JSON string in __post_init__)
        additional_args_dict = self.get_additional_args_dict()
        # Keys that are handled specially by _parse_additional_vllm_args
        special_keys = {"rope_scaling"}
        for key, value in additional_args_dict.items():
            if key in special_keys:
                continue
            if value is True:
                # Boolean flags
                command.append(f"--{key}")
            elif value is None:
                # Skip false/none values
                continue
            elif isinstance(value, (list, tuple)):
                # List values
                command.append(f"--{key}")
                command.extend([str(v) for v in value])
            else:
                # Regular key-value pairs
                command.extend([f"--{key}", str(value)])

        # Add vLLM-specific arguments that need special handling
        command.extend(self._parse_additional_vllm_args())

        return command

    def _parse_additional_vllm_args(self) -> List[str]:
        """Parse additional vLLM-specific arguments that need special handling.

        These are arguments that cannot be handled by simple string conversion
        and require special formatting (e.g., JSON serialization).

        Returns:
            List of formatted command-line arguments
        """
        args = []
        additional_args_dict = self.get_additional_args_dict()

        # Handle special cases like rope_scaling which takes JSON
        if "rope_scaling" in additional_args_dict:
            import json

            rope_config = additional_args_dict["rope_scaling"]
            rope_json = json.dumps(rope_config)
            args.extend(["--rope-scaling", rope_json])

        return args


def create_vllm_server_manager(
    model: str,
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_ids: Optional[List[int]] = None,
    **kwargs,
) -> VLLMServerManager:
    """Convenience function to create a vLLM server manager.

    Args:
        model: Model name or path
        port: Server port
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_ids: List of specific GPU IDs to use
        **kwargs: Additional configuration parameters

    Returns:
        Configured VLLMServerManager instance
    """
    config = ServerConfig(
        engine="vllm",
        model=model,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        gpu_ids=gpu_ids,
        **kwargs,
    )

    return VLLMServerManager(config)
