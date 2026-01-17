from veeksha.core.lazy_loader import _LazyLoader
from veeksha.types import SessionGraphType
from veeksha.types.base_registry import BaseRegistry


class SessionGraphGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> SessionGraphType:
        return SessionGraphType.from_str(key_str)  # type: ignore


SessionGraphGeneratorRegistry.register(
    SessionGraphType.LINEAR,
    _LazyLoader(
        "veeksha.generator.session_graph.linear",
        "LinearSessionGraphGenerator",
    ),
)

SessionGraphGeneratorRegistry.register(
    SessionGraphType.SINGLE_REQUEST,
    _LazyLoader(
        "veeksha.generator.session_graph.single_request",
        "SingleRequestSessionGraphGenerator",
    ),
)

SessionGraphGeneratorRegistry.register(
    SessionGraphType.BRANCHING,
    _LazyLoader(
        "veeksha.generator.session_graph.branching",
        "BranchingSessionGraphGenerator",
    ),
)
