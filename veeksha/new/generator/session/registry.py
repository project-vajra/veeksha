from veeksha.core.lazy_loader import _LazyLoader
from veeksha.new.types import SessionGeneratorType
from veeksha.types.base_registry import BaseRegistry


class SessionGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> SessionGeneratorType:
        return SessionGeneratorType.from_str(key_str)  # type: ignore


SessionGeneratorRegistry.register(
    SessionGeneratorType.SYNTHETIC,
    _LazyLoader(
        "veeksha.new.generator.session.synthetic",
        "SyntheticSessionGenerator",
    ),
)
SessionGeneratorRegistry.register(
    SessionGeneratorType.TRACE,
    _LazyLoader("veeksha.new.generator.session.trace", "TraceSessionGenerator"),
)
SessionGeneratorRegistry.register(
    SessionGeneratorType.LMEVAL,
    _LazyLoader(
        "veeksha.new.generator.session.lmeval",
        "LMEvalSessionGenerator",
    ),
)
