"""
Server configuration for launcher-managed inference systems.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeAlias, Union

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.types import ServerType

_DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

VLLM_OMNI_DEFAULT_IMAGE = "vllm-omni:0.21-local"
SGLANG_OMNI_DEFAULT_IMAGE = "frankleeeee/sglang-omni:dev"

SGLANG_OMNI_DEFAULT_BOOTSTRAP = """set -e
if [ ! -x {venv}/bin/sgl-omni ]; then
  uv venv {venv} -p 3.12
  . {venv}/bin/activate
  cd {src}
  uv pip install -e .
  uv pip install transformers==4.57.3 accelerate==1.12.0 sox einops
  uv pip install --no-deps qwen-tts==0.1.1
else
  . {venv}/bin/activate
fi"""


@frozen_dataclass(allow_from_file=True)
class BaseServerConfig(BasePolyConfig):
    """Base configuration for a managed inference server."""

    env_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a Python environment directory (virtualenv/conda)."},
    )

    model: str = field(
        default=_DEFAULT_MODEL,
        metadata={"help": "Model name exposed to the benchmark client."},
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

    startup_timeout: float = field(
        default=300.0, metadata={"help": "Timeout in seconds for server startup"}
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

    api_base: Optional[str] = field(
        default=None,
        metadata={"help": "External API base URL for the managed engine."},
    )
    health_url: Optional[str] = field(
        default=None,
        metadata={"help": "Health endpoint URL for the managed engine."},
    )
    setup_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Source checkout used by subprocess engines."},
    )
    max_restarts: int = field(
        default=3, metadata={"help": "Maximum managed engine restarts per sweep run."}
    )

    def __post_init__(self) -> None:
        if self.port <= 0:
            raise ValueError("server.port must be a positive integer")
        if self.startup_timeout <= 0:
            raise ValueError("server.startup_timeout must be positive")
        if self.health_check_interval <= 0:
            raise ValueError("server.health_check_interval must be positive")
        if self.max_restarts < 0:
            raise ValueError("server.max_restarts must be >= 0")
        if self.tensor_parallel_size <= 0:
            raise ValueError("server.tensor_parallel_size must be positive")
        if self.gpu_ids is not None and any(gpu_id < 0 for gpu_id in self.gpu_ids):
            raise ValueError("server.gpu_ids must contain non-negative IDs")

    def get_api_base_url(self) -> str:
        if self.api_base:
            return self.api_base.rstrip("/")
        return f"http://{self.host}:{self.port}/v1"

    def get_health_check_url(self) -> str:
        if self.health_url:
            return self.health_url
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
        engine_type = self.get_type()
        return getattr(engine_type, "name", str(engine_type)).lower()

    @property
    def type(self) -> str:
        """String discriminator used by launcher-compatible config payloads."""
        return str(self.get_type())

    @property
    def api_base_url(self) -> str:
        """Compatibility accessor for launcher-managed engines."""
        return self.get_api_base_url()

    @property
    def health_check_url(self) -> str:
        """Compatibility accessor for launcher-managed engines."""
        return self.get_health_check_url()


@frozen_dataclass
class VajraServerConfig(BaseServerConfig):
    """Vajra subprocess server config."""

    command: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def get_type(cls) -> ServerType:
        return ServerType.VAJRA

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.command:
            raise ValueError("vajra requires server.command")
        if not self.setup_dir:
            raise ValueError(
                "vajra requires server.setup_dir (the Vajra source checkout, "
                "used to record the engine git commit)"
            )

    def get_api_base_url(self) -> str:
        if self.api_base:
            return self.api_base.rstrip("/")
        return f"http://{self.host}:{self.port}"


@frozen_dataclass
class VllmServerConfig(BaseServerConfig):
    """vLLM Omni Docker server config."""

    image: str = field(default=VLLM_OMNI_DEFAULT_IMAGE)
    container_name: Optional[str] = None
    container_port: Optional[int] = None
    engine_args: List[str] = field(default_factory=list)
    volumes: List[str] = field(default_factory=list)
    docker_gpus: Optional[str] = None
    docker_runtime: Optional[str] = "nvidia"
    ipc_mode: Optional[str] = "host"
    hf_model: str = ""
    deploy_config: str = ""
    container_deploy_config: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    pass_env: List[str] = field(default_factory=list)
    bootstrap: Optional[str] = None

    @classmethod
    def get_type(cls) -> ServerType:
        return ServerType.VLLM

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_docker_engine_config(self)
        if not self.hf_model:
            raise ValueError("vllm requires server.hf_model")
        if not self.deploy_config:
            raise ValueError("vllm requires server.deploy_config")
        _set_default_client_model(self, self.hf_model)

    def get_api_base_url(self) -> str:
        if self.api_base:
            return self.api_base.rstrip("/")
        return f"http://{self.host}:{self.port}/v1"

    def get_health_check_url(self) -> str:
        if self.health_url:
            return self.health_url
        return f"{self.get_api_base_url()}/models"

    @property
    def container_name_prefix(self) -> str:
        return self.container_name or "veeksha-vllm-omni"

    @property
    def resolved_container_port(self) -> int:
        return self.container_port or self.port

    @property
    def resolved_container_deploy_config(self) -> str:
        if self.container_deploy_config:
            return self.container_deploy_config
        return f"/etc/vllm-omni/{Path(self.deploy_config).name}"

    @property
    def uses_bootstrap(self) -> bool:
        """Whether a bootstrap snippet runs before ``vllm serve``."""
        return self.resolved_bootstrap != ""

    @property
    def resolved_bootstrap(self) -> str:
        return self.bootstrap or ""


@frozen_dataclass
class SglangServerConfig(BaseServerConfig):
    """SGLang Omni Docker server config."""

    image: str = field(default=SGLANG_OMNI_DEFAULT_IMAGE)
    container_name: Optional[str] = None
    container_port: Optional[int] = None
    engine_args: List[str] = field(default_factory=list)
    volumes: List[str] = field(default_factory=list)
    docker_gpus: Optional[str] = None
    docker_runtime: Optional[str] = "nvidia"
    ipc_mode: Optional[str] = "host"
    shm_size: Optional[str] = "32g"
    docker_run_args: List[str] = field(default_factory=list)
    model_path: str = ""
    model_name: Optional[str] = None
    deploy_config: str = ""
    container_deploy_config: Optional[str] = None
    source_dir: Optional[str] = None
    container_source_dir: str = "/sglang-omni"
    venv_path: str = "/opt/sglomni/.venv"
    bootstrap: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    pass_env: List[str] = field(default_factory=list)

    @classmethod
    def get_type(cls) -> ServerType:
        return ServerType.SGLANG

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_docker_engine_config(self)
        if not self.model_path:
            raise ValueError("sglang requires server.model_path")
        if not self.deploy_config:
            raise ValueError("sglang requires server.deploy_config")
        if self.uses_bootstrap and not self.source_dir:
            raise ValueError(
                "sglang requires server.source_dir when bootstrap is enabled "
                "(the default bootstrap installs sglang-omni from the mounted "
                "source checkout); set server.bootstrap to '' to disable"
            )
        _set_default_client_model(self, self.model_name or self.model_path)

    def get_api_base_url(self) -> str:
        if self.api_base:
            return self.api_base.rstrip("/")
        return f"http://{self.host}:{self.port}/v1"

    def get_health_check_url(self) -> str:
        if self.health_url:
            return self.health_url
        return f"http://{self.host}:{self.port}/health"

    @property
    def container_name_prefix(self) -> str:
        return self.container_name or "veeksha-sglang-omni"

    @property
    def resolved_container_port(self) -> int:
        return self.container_port or self.port

    @property
    def resolved_container_deploy_config(self) -> str:
        if self.container_deploy_config:
            return self.container_deploy_config
        return f"/etc/sglang-omni/{Path(self.deploy_config).name}"

    @property
    def uses_bootstrap(self) -> bool:
        """Whether a bootstrap snippet runs before ``sgl-omni serve``."""
        return self.resolved_bootstrap != ""

    @property
    def resolved_bootstrap(self) -> str:
        if self.bootstrap is not None:
            return self.bootstrap
        return SGLANG_OMNI_DEFAULT_BOOTSTRAP.format(
            src=self.container_source_dir, venv=self.venv_path
        )


def _set_default_client_model(config: BaseServerConfig, server_model: str) -> None:
    if config.model == _DEFAULT_MODEL:
        object.__setattr__(config, "model", server_model)


def _validate_docker_engine_config(config: VllmServerConfig | SglangServerConfig) -> None:
    if config.container_port is not None and config.container_port <= 0:
        raise ValueError("server.container_port must be positive")
    if config.docker_gpus is not None and config.gpu_ids is not None:
        raise ValueError("use either server.docker_gpus or server.gpu_ids, not both")


LauncherEngineConfig: TypeAlias = VajraServerConfig | VllmServerConfig | SglangServerConfig
ManagedEngineConfig: TypeAlias = LauncherEngineConfig

# Backward-compatible import aliases for the transitional launcher names.
VajraSubprocessServerConfig = VajraServerConfig
VllmOmniDockerServerConfig = VllmServerConfig
SglangOmniDockerServerConfig = SglangServerConfig
VajraSubprocessEngineConfig = VajraServerConfig
VllmOmniDockerEngineConfig = VllmServerConfig
SglangOmniDockerEngineConfig = SglangServerConfig
