from veeksha.core.lazy_loader import _LazyLoader
from veeksha.types import SessionGeneratorType
from veeksha.types.base_registry import BaseRegistry


class SessionGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> SessionGeneratorType:
        return SessionGeneratorType.from_str(key_str)  # type: ignore


SessionGeneratorRegistry.register(
    SessionGeneratorType.SYNTHETIC,
    _LazyLoader(
        "veeksha.generator.session.synthetic",
        "SyntheticSessionGenerator",
    ),
)
SessionGeneratorRegistry.register(
    SessionGeneratorType.LMEVAL,
    _LazyLoader(
        "veeksha.generator.session.lmeval",
        "LMEvalSessionGenerator",
    ),
)
SessionGeneratorRegistry.register(
    SessionGeneratorType.TRACE,
    _LazyLoader(
        "veeksha.generator.session.trace",
        "TraceSessionGenerator",
    ),
)
