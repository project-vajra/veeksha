from typing import Any

from veeksha.types import RequestGeneratorType
from veeksha.types.base_registry import BaseRegistry
from veeksha.core.lazy_loader import _LazyLoader


class RequestGeneratorRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> RequestGeneratorType:
        return RequestGeneratorType.from_str(key_str)  # type: ignore


# Use lazy imports to avoid loading heavy dependencies (transformers, etc.) at module import time
RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC,
    _LazyLoader(
        "veeksha.generators.request_generator.synthetic_generator",
        "SyntheticRequestGenerator",
    ),
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.TRACE,
    _LazyLoader(
        "veeksha.generators.request_generator.trace_generator", "TraceRequestGenerator"
    ),
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.LMEVAL,
    _LazyLoader(
        "veeksha.generators.request_generator.lmeval_generator",
        "LMEvalRequestGenerator",
    ),
)
