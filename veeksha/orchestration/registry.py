from veeksha.core.lazy_loader import _LazyLoader
from veeksha.types import ServerType
from veeksha.types.base_registry import BaseRegistry


class ServerManagerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ServerType:
        return ServerType.from_str(key_str)  # type: ignore


ServerManagerRegistry.register(
    ServerType.VLLM,
    _LazyLoader(
        "veeksha.orchestration.managed_engines",
        "VllmOmniDockerRunner",
    ),
)
ServerManagerRegistry.register(
    ServerType.VAJRA,
    _LazyLoader(
        "veeksha.orchestration.managed_engines",
        "VajraSubprocessRunner",
    ),
)
ServerManagerRegistry.register(
    ServerType.SGLANG,
    _LazyLoader(
        "veeksha.orchestration.managed_engines",
        "SglangOmniDockerRunner",
    ),
)
