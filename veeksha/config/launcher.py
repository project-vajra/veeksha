"""Configuration model for orchestrated Veeksha launcher runs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml
from vidhi import create_class_from_dict
from vidhi.utils import get_all_subclasses

from veeksha.config.endpoint import EndpointConfig
from veeksha.config.server import BaseServerConfig, ManagedServerConfig
from veeksha.sweeps import planner as sweep_planner


class LauncherConfigError(ValueError):
    """Raised when a launcher YAML file is invalid."""


_SERVER_TYPE_ERROR = "server.type must be one of: vajra, vllm, sglang"


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
    server: Optional[ManagedServerConfig] = None
    endpoint: Optional[EndpointConfig] = None
    retry: RetryConfig = field(default_factory=RetryConfig)
    output_dir: str = field(default_factory=lambda: _default_output_dir())

    def __post_init__(self) -> None:
        if self.server is not None and self.endpoint is not None:
            raise LauncherConfigError(
                "launcher config accepts either server or endpoint, not both"
            )

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
            raw, {"server", "endpoint", "sweep", "retry", "output_dir"}, "launcher"
        )

        sweep_raw = raw.get("sweep")
        if not isinstance(sweep_raw, Mapping):
            raise LauncherConfigError("launcher config requires a sweep mapping")
        try:
            sweep = sweep_planner.SweepConfig.from_mapping(sweep_raw)
            sweep, _ = sweep_planner.resolve_sweep_config(sweep)
        except sweep_planner.SweepConfigError as exc:
            raise LauncherConfigError(str(exc)) from exc

        server_raw = raw.get("server")
        endpoint_raw = raw.get("endpoint")
        if server_raw is not None and endpoint_raw is not None:
            raise LauncherConfigError(
                "launcher config accepts either server or endpoint, not both"
            )

        server = None
        endpoint = None
        if server_raw is not None:
            if not isinstance(server_raw, Mapping):
                raise LauncherConfigError("server must be a mapping when provided")
            server = _server_config_from_mapping(server_raw)
            _validate_endpoint_matches_sweep(
                server.get_endpoint(), sweep.engine, "server.type"
            )
        if endpoint_raw is not None:
            if not isinstance(endpoint_raw, Mapping):
                raise LauncherConfigError("endpoint must be a mapping when provided")
            endpoint = _endpoint_config_from_mapping(endpoint_raw)
            _validate_endpoint_matches_sweep(
                endpoint, sweep.engine, "endpoint.engine_type"
            )

        retry_raw = raw.get("retry", {})
        if not isinstance(retry_raw, Mapping):
            raise LauncherConfigError("retry must be a mapping")
        retry = _retry_config_from_mapping(retry_raw)

        output_dir = raw.get("output_dir") or _default_output_dir()
        if not isinstance(output_dir, str):
            raise LauncherConfigError("output_dir must be a string")

        return cls(
            sweep=sweep,
            server=server,
            endpoint=endpoint,
            retry=retry,
            output_dir=output_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_data(self)


def _default_output_dir() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%d_%m_%Y-%H_%M_%S")
    return f"benchmark_output/veeksha_launcher_{timestamp}"


def _server_config_from_mapping(raw: Mapping[str, Any]) -> ManagedServerConfig:
    engine_type = raw.get("type")
    config_cls = _server_config_class_from_type(engine_type)
    data = _coerce_server_dict(raw, config_cls)
    data.pop("type", None)
    try:
        return create_class_from_dict(config_cls, data)
    except (TypeError, ValueError) as exc:
        raise LauncherConfigError(str(exc)) from exc


def _endpoint_config_from_mapping(raw: Mapping[str, Any]) -> EndpointConfig:
    _reject_unknown_keys(
        raw, {field.name for field in fields(EndpointConfig)}, "endpoint"
    )
    try:
        return create_class_from_dict(EndpointConfig, dict(raw))
    except (TypeError, ValueError) as exc:
        raise LauncherConfigError(str(exc)) from exc


def _validate_endpoint_matches_sweep(
    endpoint: EndpointConfig, sweep_engine: str, source: str
) -> None:
    if endpoint.engine_type != sweep_engine:
        raise LauncherConfigError(
            f"{source} must match sweep.engine "
            f"({source}={endpoint.engine_type}, sweep.engine={sweep_engine})"
        )


def _server_config_class_from_type(engine_type: object) -> type[BaseServerConfig]:
    if not isinstance(engine_type, str):
        raise LauncherConfigError(_SERVER_TYPE_ERROR)
    normalized_engine_type = engine_type.lower()
    for subclass in get_all_subclasses(BaseServerConfig):
        try:
            subtype = subclass.get_type()
        except NotImplementedError:
            continue
        if getattr(subtype, "name", "").lower() == normalized_engine_type:
            return subclass
        if str(subtype).lower() == normalized_engine_type:
            return subclass
    raise LauncherConfigError(_SERVER_TYPE_ERROR)


def _coerce_server_dict(
    raw: Mapping[str, Any], config_cls: type[BaseServerConfig]
) -> dict[str, Any]:
    allowed = {field.name for field in fields(config_cls)} | {"type"}
    _reject_unknown_keys(raw, allowed, "server")
    data = dict(raw)

    for key in ("host", "type", "api_key", "dtype"):
        _coerce_present_str(data, key)
    for key in (
        "api_base",
        "health_url",
        "setup_dir",
        "env_path",
        "max_model_len",
        "client_provider",
    ):
        if key == "max_model_len":
            _coerce_optional_int(data, key)
        else:
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
        "model",
    ):
        _coerce_optional_str(data, key)
    _coerce_present_int(data, "port")
    for key in ("max_restarts", "tensor_parallel_size"):
        _coerce_default_int(data, key)
    _coerce_optional_int(data, "container_port")
    for key in ("startup_timeout", "health_check_interval"):
        _coerce_default_float(data, key)
    for key in ("command", "engine_args", "volumes", "pass_env", "docker_run_args"):
        _coerce_string_list(data, key)
    _coerce_string_mapping(data, "env")
    _coerce_int_list(data, "gpu_ids")
    _coerce_bool(data, "require_contiguous_gpus")
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
        raise LauncherConfigError(f"server.{key} must be a string")


def _coerce_optional_str(data: dict[str, Any], key: str) -> None:
    if key not in data or data[key] is None:
        return
    if not isinstance(data[key], str):
        raise LauncherConfigError(f"server.{key} must be a string")


def _coerce_default_str(data: dict[str, Any], key: str) -> None:
    if data.get(key) is None:
        data.pop(key, None)
        return
    if not isinstance(data[key], str):
        raise LauncherConfigError(f"server.{key} must be a string")


def _coerce_present_int(data: dict[str, Any], key: str) -> None:
    if key not in data:
        return
    if isinstance(data[key], bool) or not isinstance(data[key], int):
        raise LauncherConfigError(f"server.{key} must be an integer")


def _coerce_optional_int(data: dict[str, Any], key: str) -> None:
    if key not in data or data[key] is None:
        return
    if isinstance(data[key], bool) or not isinstance(data[key], int):
        raise LauncherConfigError(f"server.{key} must be an integer")


def _coerce_default_int(data: dict[str, Any], key: str) -> None:
    if data.get(key) is None:
        data.pop(key, None)
        return
    if isinstance(data[key], bool) or not isinstance(data[key], int):
        raise LauncherConfigError(f"server.{key} must be an integer")


def _coerce_default_float(data: dict[str, Any], key: str) -> None:
    if data.get(key) is None:
        data.pop(key, None)
        return
    if isinstance(data[key], bool) or not isinstance(data[key], (int, float)):
        raise LauncherConfigError(f"server.{key} must be a number")
    data[key] = float(data[key])


def _coerce_string_list(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if value is None:
        data.pop(key, None)
        return
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LauncherConfigError(f"server.{key} must be a list of strings")


def _coerce_int_list(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if value is None:
        return
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise LauncherConfigError(f"server.{key} must be a list of integers")


def _coerce_bool(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if value is None:
        return
    if not isinstance(value, bool):
        raise LauncherConfigError(f"server.{key} must be a boolean")


def _coerce_string_mapping(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if value is None:
        data.pop(key, None)
        return
    if not isinstance(value, Mapping):
        raise LauncherConfigError(f"server.{key} must be a mapping")
    data[key] = {str(k): str(v) for k, v in value.items()}


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        data = {
            field.name: _to_plain_data(getattr(value, field.name))
            for field in fields(value)
        }
        if hasattr(value, "get_type") and callable(getattr(value, "get_type")):
            data["type"] = str(value.get_type())
        return data
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    return value
