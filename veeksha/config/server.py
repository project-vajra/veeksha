"""
Server configuration for LLM inference systems.

This module provides configuration classes for launching and managing
LLM inference servers like vLLM.
"""

import json
from dataclasses import field
from typing import Any, Dict, List, Optional, Union

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class ServerConfig:
    """Configuration for server launch and management.

    This class defines all parameters needed to launch, manage, and
    connect to an LLM inference server.
    """

    # Server identification
    engine: str = field(
        default="vllm",
        metadata={
            "help": "The inference engine to use (e.g., 'vajra', 'vllm','tgi', 'sglang')"
        },
    )

    environment_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a Python environment directory (virtualenv/conda). "
            "If provided, the environment's bin/Scripts directory will be prepended to PATH "
            "when launching servers. If None, the current PATH is used."
        },
    )

    # Model configuration
    # Note: When used with BenchmarkConfig, this will be auto-populated from ClientConfig
    model: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Model name or path"},
    )

    # Server connection
    host: str = field(
        default="localhost", metadata={"help": "Host address for the server"}
    )

    port: int = field(default=8000, metadata={"help": "Port number for the server"})

    api_key: str = field(
        default="token-abc123", metadata={"help": "API key for server authentication"}
    )

    # Hardware/Resource configuration
    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Number of GPUs for tensor parallelism"}
    )

    gpu_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of GPU IDs to use (None means auto-assign)"},
    )

    # Server-specific arguments
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type for model weights (auto, float16, bfloat16, etc.)"
        },
    )

    max_model_len: Optional[int] = field(
        default=None, metadata={"help": "Maximum model context length"}
    )

    additional_args: Union[str, Dict[str, Any], None] = field(
        default="{}",
        metadata={
            "help": "Additional engine-specific arguments as JSON string, dict, or None. "
            'Example: \'{"enable-prefix-caching": true}\' or {"enable-prefix-caching": true}'
        },
    )

    # Startup configuration
    startup_timeout: int = field(
        default=300, metadata={"help": "Timeout in seconds for server startup"}
    )

    health_check_interval: float = field(
        default=2.0, metadata={"help": "Interval in seconds between health checks"}
    )

    # Lifecycle management
    auto_shutdown: bool = field(
        default=True, metadata={"help": "Automatically shutdown server after benchmark"}
    )

    # Resource management
    require_contiguous_gpus: bool = field(
        default=True,
        metadata={
            "help": "Require contiguous GPU allocation (e.g., GPUs 0,1,2 instead of 0,2,5)"
        },
    )

    def get_api_base_url(self) -> str:
        """Get the full API base URL."""
        return f"http://{self.host}:{self.port}/v1"

    def get_health_check_url(self) -> str:
        """Get the health check endpoint URL."""
        return f"http://{self.host}:{self.port}/health"

    def get_gpu_env_var(self) -> Optional[str]:
        """Get CUDA_VISIBLE_DEVICES value if gpu_ids is specified."""
        if self.gpu_ids is not None:
            return ",".join(map(str, self.gpu_ids))
        return None

    def get_num_gpus(self) -> int:
        """Get the number of GPUs required for this server.

        Returns:
            Number of GPUs (tensor_parallel_size or length of gpu_ids)
        """
        if self.gpu_ids is not None:
            return len(self.gpu_ids)
        return self.tensor_parallel_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        # Parse additional_args
        additional_args_dict: Dict[str, Any] = {}
        if self.additional_args is None:
            additional_args_dict = {}
        elif isinstance(self.additional_args, dict):
            additional_args_dict = self.additional_args
        elif isinstance(self.additional_args, str):
            try:
                additional_args_dict = json.loads(self.additional_args)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(
                    f"Invalid JSON in configuration field 'additional_args': {self.additional_args[:100]}{'...' if len(self.additional_args) > 100 else ''}. Original error: {e}"
                ) from e
        else:
            raise TypeError(
                f"additional_args must be None, dict, or str (JSON), got {type(self.additional_args).__name__}: {self.additional_args!r}"
            )

        return {
            "engine": self.engine,
            "model": self.model if isinstance(self.model, str) else str(self.model),
            "environment_path": self.environment_path,
            "host": self.host,
            "port": self.port,
            "api_key": self.api_key,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_ids": self.gpu_ids,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "additional_args": additional_args_dict,  # Use parsed dict
            "startup_timeout": self.startup_timeout,
            "health_check_interval": self.health_check_interval,
            "auto_shutdown": self.auto_shutdown,
            "require_contiguous_gpus": self.require_contiguous_gpus,
        }
