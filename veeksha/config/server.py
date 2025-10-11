"""
Server configuration for LLM inference systems.

This module provides configuration classes for launching and managing
LLM inference servers like vLLM.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ServerConfig:
    """Configuration for server launch and management.

    This class defines all parameters needed to launch, manage, and
    connect to an LLM inference server.
    """

    # Server identification
    engine: str = field(
        default="vllm",
        metadata={
            "help": "The inference engine to use (e.g., 'vllm', 'tgi', 'sglang')"
        },
    )

    # Model configuration
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

    additional_args: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Additional engine-specific arguments"}
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

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "engine": self.engine,
            "model": self.model,
            "host": self.host,
            "port": self.port,
            "api_key": self.api_key,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_ids": self.gpu_ids,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "additional_args": self.additional_args,
            "startup_timeout": self.startup_timeout,
            "health_check_interval": self.health_check_interval,
            "auto_shutdown": self.auto_shutdown,
        }
