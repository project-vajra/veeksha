"""
Server configuration for LLM inference systems.
"""

from dataclasses import field
from typing import Any, Dict, List, Optional, Union

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.types import ServerType


@frozen_dataclass(allow_from_file=True)
class BaseServerConfig(BasePolyConfig):
    """Base configuration for server launch and management."""

    env_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a Python environment directory (virtualenv/conda)."},
    )

    model: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Model name or path."},
    )

    host: str = field(
        default="localhost", metadata={"help": "Host address for the server"}
    )

    port: int = field(default=8000, metadata={"help": "Port number for the server"})

    api_key: str = field(
        default="token-abc123", metadata={"help": "API key for server authentication"}
    )

    gpu_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of GPU IDs to use (None means auto-assign)"},
    )

    startup_timeout: int = field(
        default=300, metadata={"help": "Timeout in seconds for server startup"}
    )

    health_check_interval: float = field(
        default=2.0, metadata={"help": "Interval in seconds between health checks"}
    )

    require_contiguous_gpus: bool = field(
        default=True,
        metadata={
            "help": "Require contiguous GPU allocation (e.g., GPUs 0,1,2 instead of 0,2,5)"
        },
    )

    # engine arguments

    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Number of GPUs for tensor parallelism"}
    )

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
            "help": "Additional engine-specific arguments as JSON string, dict, or None."
        },
    )

    def get_api_base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def get_health_check_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def get_gpu_env_var(self) -> Optional[str]:
        """Get CUDA_VISIBLE_DEVICES value if gpu_ids is specified."""
        if self.gpu_ids is not None:
            return ",".join(map(str, self.gpu_ids))
        return None

    def get_num_gpus(self) -> int:
        """Get the number of GPUs required for this server."""
        if self.gpu_ids is not None:
            return len(self.gpu_ids)
        return self.tensor_parallel_size

    @property
    def engine(self) -> str:
        """Get the engine name for logging/compat."""
        return self.get_type().name.lower()


@frozen_dataclass
class VllmServerConfig(BaseServerConfig):
    @classmethod
    def get_type(cls) -> ServerType:
        return ServerType.VLLM


@frozen_dataclass
class VajraServerConfig(BaseServerConfig):
    @classmethod
    def get_type(cls) -> ServerType:
        return ServerType.VAJRA


@frozen_dataclass
class SglangServerConfig(BaseServerConfig):
    @classmethod
    def get_type(cls) -> ServerType:
        return ServerType.SGLANG
