"""Evaluator registry for the new veeksha framework."""

from veeksha.core.lazy_loader import _LazyLoader
from veeksha.types import ChannelModality, EvaluationType
from veeksha.types.base_registry import BaseRegistry


class EvaluatorRegistry(BaseRegistry):
    """Registry for evaluator implementations."""

    @classmethod
    def get_key_from_str(cls, key_str: str) -> EvaluationType:
        return EvaluationType.from_str(key_str)  # type: ignore


EvaluatorRegistry.register(
    EvaluationType.PERFORMANCE,
    _LazyLoader(
        "veeksha.evaluator.performance.base",
        "PerformanceEvaluator",
    ),
)

EvaluatorRegistry.register(
    EvaluationType.ACCURACY_LMEVAL,
    _LazyLoader(
        "veeksha.evaluator.accuracy.base",
        "LMEvalAccuracyEvaluator",
    ),
)


class ChannelPerformanceEvaluatorRegistry(BaseRegistry):
    """Registry for channel-specific performance evaluators."""

    @classmethod
    def get_key_from_str(cls, key_str: str) -> ChannelModality:
        return ChannelModality.from_str(key_str)  # type: ignore


ChannelPerformanceEvaluatorRegistry.register(
    ChannelModality.TEXT,
    _LazyLoader(
        "veeksha.evaluator.performance.text",
        "TextPerformanceEvaluator",
    ),
)

ChannelPerformanceEvaluatorRegistry.register(
    ChannelModality.IMAGE,
    _LazyLoader(
        "veeksha.evaluator.performance.image",
        "ImagePerformanceEvaluator",
    ),
)

ChannelPerformanceEvaluatorRegistry.register(
    ChannelModality.AUDIO,
    _LazyLoader(
        "veeksha.evaluator.performance.audio",
        "AudioPerformanceEvaluator",
    ),
)

ChannelPerformanceEvaluatorRegistry.register(
    ChannelModality.VIDEO,
    _LazyLoader(
        "veeksha.evaluator.performance.video",
        "VideoPerformanceEvaluator",
    ),
)
