"""Inference endpoint contract shared by launchers and benchmarks."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from typing import Any, Optional

from vidhi import field, frozen_dataclass

from veeksha.types import ServerType


@frozen_dataclass
class EndpointConfig:
    """Client-facing contract for an inference endpoint."""

    engine_type: str = field("", help="Endpoint engine type: vajra, vllm, or sglang.")
    model: str = field("", help="Model name exposed by the endpoint.")
    api_base: str = field("", help="OpenAI-compatible API base URL.")
    api_key: Optional[str] = field(None, help="API key for the endpoint.")
    health_url: Optional[str] = field(None, help="Health endpoint URL.")
    host: str = field("localhost", help="Host used for local port ownership checks.")
    port: int = field(0, help="Host port owned by the endpoint.")

    def __post_init__(self) -> None:
        self.engine_type = _normalize_engine_type(self.engine_type)
        if not self.model:
            raise ValueError("endpoint.model must be non-empty")
        if not self.api_base:
            raise ValueError("endpoint.api_base must be non-empty")
        self.api_base = self.api_base.rstrip("/")
        if self.health_url is not None:
            self.health_url = self.health_url.rstrip("/")
        if not self.host:
            raise ValueError("endpoint.host must be non-empty")
        if self.port < 0:
            raise ValueError("endpoint.port must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def client_overrides_for(self, client_config: Any) -> dict[str, Any]:
        client_fields = {config_field.name for config_field in fields(client_config)}
        overrides: dict[str, Any] = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "model": self.model,
        }
        return {
            key: value
            for key, value in overrides.items()
            if key in client_fields and value is not None
        }

    def apply_to_client_config(self, client_config: Any) -> Any:
        return replace(client_config, **self.client_overrides_for(client_config))

    def apply_to_client_mapping(self, client_mapping: dict[str, Any]) -> None:
        client_mapping["api_base"] = self.api_base
        client_mapping["model"] = self.model
        if self.api_key is not None:
            client_mapping["api_key"] = self.api_key


def _normalize_engine_type(engine_type: str | ServerType) -> str:
    if isinstance(engine_type, ServerType):
        return str(engine_type)
    if not isinstance(engine_type, str) or not engine_type:
        raise ValueError("endpoint.engine_type must be one of: vajra, vllm, sglang")
    try:
        return str(ServerType.from_str(engine_type))
    except KeyError as exc:
        raise ValueError(
            "endpoint.engine_type must be one of: vajra, vllm, sglang"
        ) from exc
