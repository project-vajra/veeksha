from veeksha.core.lazy_loader import _LazyLoader
from veeksha.types import TraceFlavorType
from veeksha.types.base_registry import BaseRegistry


class TraceFlavorGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> TraceFlavorType:
        return TraceFlavorType.from_str(key_str)  # type: ignore


TraceFlavorGeneratorRegistry.register(
    TraceFlavorType.TIMED_SYNTHETIC_MULTI_TURN,
    _LazyLoader(
        "veeksha.generator.session.trace.multi_turn",
        "TimedSyntheticMultiTurnTraceFlavorGenerator",
    ),
)
TraceFlavorGeneratorRegistry.register(
    TraceFlavorType.SHARED_PREFIX,
    _LazyLoader(
        "veeksha.generator.session.trace.shared_prefix",
        "SharedPrefixTraceFlavorGenerator",
    ),
)
TraceFlavorGeneratorRegistry.register(
    TraceFlavorType.RAG,
    _LazyLoader(
        "veeksha.generator.session.trace.rag",
        "RAGTraceFlavorGenerator",
    ),
)
TraceFlavorGeneratorRegistry.register(
    TraceFlavorType.REQUEST_LOG,
    _LazyLoader(
        "veeksha.generator.session.trace.request_log",
        "RequestLogTraceFlavorGenerator",
    ),
)
TraceFlavorGeneratorRegistry.register(
    TraceFlavorType.UNTIMED_CONTENT_MULTI_TURN,
    _LazyLoader(
        "veeksha.generator.session.trace.conversation",
        "UntimedContentMultiTurnTraceFlavorGenerator",
    ),
)
