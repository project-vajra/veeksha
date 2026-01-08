from veeksha.core.lazy_loader import _LazyLoader
from veeksha.types import TraceFlavorType
from veeksha.types.base_registry import BaseRegistry


class TraceFlavorGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> TraceFlavorType:
        return TraceFlavorType.from_str(key_str)  # type: ignore


TraceFlavorGeneratorRegistry.register(
    TraceFlavorType.CLAUDE_CODE,
    _LazyLoader(
        "veeksha.generator.session.trace.claude_code",
        "ClaudeCodeTraceFlavorGenerator",
    ),
)
TraceFlavorGeneratorRegistry.register(
    TraceFlavorType.MOONCAKE_CONV,
    _LazyLoader(
        "veeksha.generator.session.trace.mooncake_conv",
        "MooncakeConvTraceFlavorGenerator",
    ),
)
TraceFlavorGeneratorRegistry.register(
    TraceFlavorType.RAG,
    _LazyLoader(
        "veeksha.generator.session.trace.rag",
        "RAGTraceFlavorGenerator",
    ),
)
