"""Configuration model for the orchestrated Veeksha launcher."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, TypeAlias

import yaml

from veeksha.sweeps import planner as sweep_planner


class LauncherConfigError(ValueError):
    """Raised when a launcher YAML file is invalid."""


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


@dataclass(frozen=True)
class ManagedEngineConfig:
    host: str = "localhost"
    port: int = 0
    api_base: Optional[str] = None
    health_url: Optional[str] = None
    setup_dir: Optional[str] = None
    startup_timeout: float = 600.0
    health_check_interval: float = 2.0
    max_restarts: int = 3

    def __post_init__(self) -> None:
        if self.port <= 0:
            raise LauncherConfigError("engine.port must be a positive integer")
        if self.startup_timeout <= 0:
            raise LauncherConfigError("engine.startup_timeout must be positive")
        if self.health_check_interval <= 0:
            raise LauncherConfigError("engine.health_check_interval must be positive")
        if self.max_restarts < 0:
            raise LauncherConfigError("engine.max_restarts must be >= 0")

    @property
    def api_base_url(self) -> str:
        if self.api_base:
            return self.api_base.rstrip("/")
        return f"http://{self.host}:{self.port}"

    @property
    def health_check_url(self) -> str:
        if self.health_url:
            return self.health_url
        return f"{self.api_base_url}/health"


@dataclass(frozen=True)
class VajraSubprocessEngineConfig(ManagedEngineConfig):
    type: str = "vajra_subprocess"
    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    gpu_ids: Optional[list[int]] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.type != "vajra_subprocess":
            raise LauncherConfigError(
                "vajra subprocess config has the wrong engine.type"
            )
        if not self.command:
            raise LauncherConfigError("vajra_subprocess requires engine.command")
        if not self.setup_dir:
            raise LauncherConfigError(
                "vajra_subprocess requires engine.setup_dir (the Vajra source "
                "checkout, used to record the engine git commit)"
            )
        if self.gpu_ids is not None and any(gpu_id < 0 for gpu_id in self.gpu_ids):
            raise LauncherConfigError("engine.gpu_ids must contain non-negative IDs")


@dataclass(frozen=True)
class VllmOmniDockerEngineConfig(ManagedEngineConfig):
    type: str = "vllm_omni_docker"
    image: str = VLLM_OMNI_DEFAULT_IMAGE
    container_name: Optional[str] = None
    container_port: Optional[int] = None
    engine_args: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)
    docker_gpus: Optional[str] = None
    docker_runtime: Optional[str] = "nvidia"
    ipc_mode: Optional[str] = "host"
    hf_model: str = ""
    deploy_config: str = ""
    container_deploy_config: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)
    gpu_ids: Optional[list[int]] = None
    pass_env: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.type != "vllm_omni_docker":
            raise LauncherConfigError(
                "vLLM Omni Docker config has the wrong engine.type"
            )
        if self.container_port is not None and self.container_port <= 0:
            raise LauncherConfigError("engine.container_port must be positive")
        if self.docker_gpus is not None and self.gpu_ids is not None:
            raise LauncherConfigError(
                "use either engine.docker_gpus or engine.gpu_ids, not both"
            )
        if self.gpu_ids is not None and any(gpu_id < 0 for gpu_id in self.gpu_ids):
            raise LauncherConfigError("engine.gpu_ids must contain non-negative IDs")
        if not self.hf_model:
            raise LauncherConfigError("vllm_omni_docker requires engine.hf_model")
        if not self.deploy_config:
            raise LauncherConfigError("vllm_omni_docker requires engine.deploy_config")

    @property
    def api_base_url(self) -> str:
        if self.api_base:
            return self.api_base.rstrip("/")
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_check_url(self) -> str:
        if self.health_url:
            return self.health_url
        return f"{self.api_base_url}/models"

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


@dataclass(frozen=True)
class SglangOmniDockerEngineConfig(ManagedEngineConfig):
    type: str = "sglang_omni_docker"
    image: str = SGLANG_OMNI_DEFAULT_IMAGE
    container_name: Optional[str] = None
    container_port: Optional[int] = None
    engine_args: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)
    docker_gpus: Optional[str] = None
    docker_runtime: Optional[str] = "nvidia"
    ipc_mode: Optional[str] = "host"
    shm_size: Optional[str] = "32g"
    docker_run_args: list[str] = field(default_factory=list)
    model_path: str = ""
    model_name: Optional[str] = None
    deploy_config: str = ""
    container_deploy_config: Optional[str] = None
    source_dir: Optional[str] = None
    container_source_dir: str = "/sglang-omni"
    venv_path: str = "/opt/sglomni/.venv"
    bootstrap: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)
    gpu_ids: Optional[list[int]] = None
    pass_env: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.type != "sglang_omni_docker":
            raise LauncherConfigError(
                "sglang Omni Docker config has the wrong engine.type"
            )
        if self.container_port is not None and self.container_port <= 0:
            raise LauncherConfigError("engine.container_port must be positive")
        if self.docker_gpus is not None and self.gpu_ids is not None:
            raise LauncherConfigError(
                "use either engine.docker_gpus or engine.gpu_ids, not both"
            )
        if self.gpu_ids is not None and any(gpu_id < 0 for gpu_id in self.gpu_ids):
            raise LauncherConfigError("engine.gpu_ids must contain non-negative IDs")
        if not self.model_path:
            raise LauncherConfigError("sglang_omni_docker requires engine.model_path")
        if not self.deploy_config:
            raise LauncherConfigError(
                "sglang_omni_docker requires engine.deploy_config"
            )
        if self.uses_bootstrap and not self.source_dir:
            raise LauncherConfigError(
                "sglang_omni_docker requires engine.source_dir when bootstrap "
                "is enabled (the default bootstrap installs sglang-omni from the "
                "mounted source checkout); set engine.bootstrap to '' to disable"
            )

    @property
    def api_base_url(self) -> str:
        if self.api_base:
            return self.api_base.rstrip("/")
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_check_url(self) -> str:
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
        """Whether a bootstrap snippet runs before ``sgl-omni serve``.

        ``bootstrap is None`` means "use the default bootstrap"; an explicit
        empty string disables it (for images that already ship ``sgl-omni``).
        """
        return self.resolved_bootstrap != ""

    @property
    def resolved_bootstrap(self) -> str:
        if self.bootstrap is not None:
            return self.bootstrap
        return SGLANG_OMNI_DEFAULT_BOOTSTRAP.format(
            src=self.container_source_dir, venv=self.venv_path
        )


LauncherEngineConfig: TypeAlias = (
    VajraSubprocessEngineConfig
    | VllmOmniDockerEngineConfig
    | SglangOmniDockerEngineConfig
)


@dataclass(frozen=True)
class RetryConfig:
    max_attempts_per_run: int = 2
    restart_engine_before_retry: bool = True
    fail_sweep_after_exhausted_retries: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts_per_run <= 0:
            raise LauncherConfigError("retry.max_attempts_per_run must be positive")


@dataclass(frozen=True)
class LauncherConfig:
    sweep: sweep_planner.SweepConfig
    engine: Optional[LauncherEngineConfig] = None
    retry: RetryConfig = field(default_factory=RetryConfig)
    output_dir: str = field(default_factory=lambda: _default_output_dir())

    @classmethod
    def from_file(cls, path: str | Path) -> "LauncherConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, Mapping):
            raise LauncherConfigError("launcher config must be a YAML mapping")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LauncherConfig":
        _reject_unknown_keys(
            raw, {"engine", "sweep", "retry", "output_dir"}, "launcher"
        )

        sweep_raw = raw.get("sweep")
        if not isinstance(sweep_raw, Mapping):
            raise LauncherConfigError("launcher config requires a sweep mapping")
        try:
            sweep = sweep_planner.SweepConfig.from_mapping(sweep_raw)
            sweep_planner.resolve_sweep_config(sweep)
        except sweep_planner.SweepConfigError as exc:
            raise LauncherConfigError(str(exc)) from exc

        engine_raw = raw.get("engine")
        engine = None
        if engine_raw is not None:
            if not isinstance(engine_raw, Mapping):
                raise LauncherConfigError("engine must be a mapping when provided")
            engine = _engine_config_from_mapping(engine_raw)

        retry_raw = raw.get("retry", {})
        if not isinstance(retry_raw, Mapping):
            raise LauncherConfigError("retry must be a mapping")
        retry = _retry_config_from_mapping(retry_raw)

        output_dir = raw.get("output_dir") or _default_output_dir()
        if not isinstance(output_dir, str):
            raise LauncherConfigError("output_dir must be a string")

        return cls(sweep=sweep, engine=engine, retry=retry, output_dir=output_dir)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_data(self)


def _default_output_dir() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%d_%m_%Y-%H_%M_%S")
    return f"benchmark_output/veeksha_launcher_{timestamp}"


def _engine_config_from_mapping(raw: Mapping[str, Any]) -> LauncherEngineConfig:
    engine_type = raw.get("type")
    if engine_type == "vajra_subprocess":
        data = _coerce_engine_dict(raw, VajraSubprocessEngineConfig)
        return VajraSubprocessEngineConfig(**data)
    if engine_type == "vllm_omni_docker":
        data = _coerce_engine_dict(raw, VllmOmniDockerEngineConfig)
        return VllmOmniDockerEngineConfig(**data)
    if engine_type == "sglang_omni_docker":
        data = _coerce_engine_dict(raw, SglangOmniDockerEngineConfig)
        return SglangOmniDockerEngineConfig(**data)
    raise LauncherConfigError(
        "engine.type must be one of: vajra_subprocess, vllm_omni_docker, "
        "sglang_omni_docker"
    )


def _coerce_engine_dict(
    raw: Mapping[str, Any], config_cls: type[ManagedEngineConfig]
) -> dict[str, Any]:
    _reject_unknown_keys(raw, {field.name for field in fields(config_cls)}, "engine")
    data = dict(raw)

    for key in ("host", "type"):
        _coerce_present_str(data, key)
    for key in ("api_base", "health_url", "setup_dir"):
        _coerce_optional_str(data, key)
    for key in ("image", "hf_model", "model_path", "deploy_config"):
        _coerce_default_str(data, key)
    for key in ("container_name", "docker_gpus", "docker_runtime", "ipc_mode"):
        _coerce_optional_str(data, key)
    for key in (
        "container_deploy_config",
        "shm_size",
        "model_name",
        "source_dir",
        "container_source_dir",
        "venv_path",
        "bootstrap",
    ):
        _coerce_optional_str(data, key)
    _coerce_present_int(data, "port")
    _coerce_default_int(data, "max_restarts")
    _coerce_optional_int(data, "container_port")
    for key in ("startup_timeout", "health_check_interval"):
        _coerce_default_float(data, key)
    for key in ("command", "engine_args", "volumes", "pass_env", "docker_run_args"):
        _coerce_string_list(data, key)
    _coerce_string_mapping(data, "env")
    _coerce_int_list(data, "gpu_ids")
    return data


def _retry_config_from_mapping(raw: Mapping[str, Any]) -> RetryConfig:
    _reject_unknown_keys(
        raw,
        {
            "max_attempts_per_run",
            "restart_engine_before_retry",
            "fail_sweep_after_exhausted_retries",
        },
        "retry",
    )
    data = dict(raw)
    value = data.get("max_attempts_per_run")
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise LauncherConfigError("retry.max_attempts_per_run must be an integer")
    for key in ("restart_engine_before_retry", "fail_sweep_after_exhausted_retries"):
        value = data.get(key)
        if value is not None and not isinstance(value, bool):
            raise LauncherConfigError(f"retry.{key} must be a boolean")
    return RetryConfig(**data)


def _reject_unknown_keys(
    raw: Mapping[str, Any], allowed: set[str], section: str
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise LauncherConfigError(
            f"unsupported {section} config keys: {', '.join(unknown)}"
        )


def _coerce_present_str(data: dict[str, Any], key: str) -> None:
    if key not in data:
        return
    if not isinstance(data[key], str):
        raise LauncherConfigError(f"engine.{key} must be a string")


def _coerce_optional_str(data: dict[str, Any], key: str) -> None:
    if key not in data or data[key] is None:
        return
    if not isinstance(data[key], str):
        raise LauncherConfigError(f"engine.{key} must be a string")


def _coerce_default_str(data: dict[str, Any], key: str) -> None:
    if data.get(key) is None:
        data.pop(key, None)
        return
    if not isinstance(data[key], str):
        raise LauncherConfigError(f"engine.{key} must be a string")


def _coerce_present_int(data: dict[str, Any], key: str) -> None:
    if key not in data:
        return
    if isinstance(data[key], bool) or not isinstance(data[key], int):
        raise LauncherConfigError(f"engine.{key} must be an integer")


def _coerce_optional_int(data: dict[str, Any], key: str) -> None:
    if key not in data or data[key] is None:
        return
    if isinstance(data[key], bool) or not isinstance(data[key], int):
        raise LauncherConfigError(f"engine.{key} must be an integer")


def _coerce_default_int(data: dict[str, Any], key: str) -> None:
    if data.get(key) is None:
        data.pop(key, None)
        return
    if isinstance(data[key], bool) or not isinstance(data[key], int):
        raise LauncherConfigError(f"engine.{key} must be an integer")


def _coerce_default_float(data: dict[str, Any], key: str) -> None:
    if data.get(key) is None:
        data.pop(key, None)
        return
    if isinstance(data[key], bool) or not isinstance(data[key], (int, float)):
        raise LauncherConfigError(f"engine.{key} must be a number")
    data[key] = float(data[key])


def _coerce_string_list(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if value is None:
        data.pop(key, None)
        return
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LauncherConfigError(f"engine.{key} must be a list of strings")


def _coerce_int_list(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if value is None:
        return
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise LauncherConfigError(f"engine.{key} must be a list of integers")


def _coerce_string_mapping(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if value is None:
        data.pop(key, None)
        return
    if not isinstance(value, Mapping):
        raise LauncherConfigError(f"engine.{key} must be a mapping")
    data[key] = {str(k): str(v) for k, v in value.items()}


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_plain_data(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    return value
