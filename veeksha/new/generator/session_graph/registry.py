from veeksha.core.lazy_loader import _LazyLoader
from veeksha.new.types import SessionGraphType
from veeksha.types.base_registry import BaseRegistry


class SessionGraphGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> SessionGraphType:
        return SessionGraphType.from_str(key_str)  # type: ignore


SessionGraphGeneratorRegistry.register(
    SessionGraphType.LINEAR,
    _LazyLoader(
        "veeksha.new.generator.session_graph.linear",
        "LinearSessionGraphGenerator",
    ),
)
