"""
SGLang server manager for orchestrating SGLang inference servers.

This module provides a concrete implementation of ServerManager for SGLang,
handling SGLang-specific launch commands and configuration.
"""

from typing import List, Optional

from veeksha.config.server import ServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration.server_manager import BaseServerManager

logger = init_logger(__name__)


class SGLangServerManager(BaseServerManager):
    """Manager for SGLang inference servers.

    This class handles launching and managing SGLang servers with
    proper command-line arguments and configuration.
    """

    def __init__(self, config: ServerConfig):
        """Initialize SGLang server manager.

        Args:
            config: Server configuration
        """
        super().__init__(config)

        if config.engine.lower() != "sglang":
            logger.warning(
                f"SGLangServerManager created with engine='{config.engine}'. "
                "Expected 'sglang'"
            )

    def _build_launch_command(self) -> List[str]:
        """Build the SGLang server launch command.

        Returns:
            List of command arguments
        """
        import sys

        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.config.model,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
        ]

        # Add tensor parallelism
        if self.config.tensor_parallel_size > 1:
            command.extend(
                ["--tensor-parallel-size", str(self.config.tensor_parallel_size)]
            )

        # Add dtype if specified
        if self.config.dtype:
            command.extend(["--dtype", self.config.dtype])

        # Add max model length if specified (context-length in sglang)
        if self.config.max_model_len is not None:
            command.extend(["--context-length", str(self.config.max_model_len)])

        # Process additional arguments (parsed from JSON string in __post_init__)
        additional_args_dict = self.get_additional_args_dict()
        for key, value in additional_args_dict.items():
            if value is True:
                # Boolean flags
                command.append(f"--{key}")
            elif value is False or value is None:
                # Skip false/none values
                continue
            elif isinstance(value, (list, tuple)):
                # List values
                command.append(f"--{key}")
                command.extend([str(v) for v in value])
            else:
                # Regular key-value pairs
                command.extend([f"--{key}", str(value)])

        # Add SGLang-specific arguments that need special handling
        command.extend(self._parse_additional_sglang_args())

        return command

    def _parse_additional_sglang_args(self) -> List[str]:
        """Parse additional SGLang-specific arguments that need special handling.

        These are arguments that cannot be handled by simple string conversion
        and require special formatting.

        Returns:
            List of formatted command-line arguments
        """
        args = []
        additional_args_dict = self.get_additional_args_dict()

        # Handle special cases if any (similar to vLLM's rope_scaling)
        # For now, no special handling needed, but this can be extended

        return args


def create_sglang_server_manager(
    model: str,
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_ids: Optional[List[int]] = None,
    **kwargs,
) -> SGLangServerManager:
    """Convenience function to create a SGLang server manager.

    Args:
        model: Model name or path
        port: Server port
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_ids: List of specific GPU IDs to use
        **kwargs: Additional configuration parameters

    Returns:
        Configured SGLangServerManager instance
    """
    config = ServerConfig(
        engine="sglang",
        model=model,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        gpu_ids=gpu_ids,
        **kwargs,
    )

    return SGLangServerManager(config)
