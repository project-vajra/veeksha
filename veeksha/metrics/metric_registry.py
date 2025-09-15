from typing import Any, Dict, List, Protocol


class MetricSpec(Protocol):
    key: str
    lower_is_better: bool

    def extract(self, request_metrics: Dict[str, Any]) -> List[float]:
        ...


class _BaseMetricSpec:
    def __init__(self, key: str, lower_is_better: bool = True) -> None:
        self.key = key
        self.lower_is_better = lower_is_better

    def extract(self, request_metrics: Dict[str, Any]) -> List[float]:  # type: ignore[override]
        values = request_metrics.get(self.key, [])
        return values if isinstance(values, list) else []


class _TtftMetricSpec(_BaseMetricSpec):
    def __init__(self) -> None:
        super().__init__(key="ttft", lower_is_better=True)


class _TpotMetricSpec(_BaseMetricSpec):
    def __init__(self) -> None:
        super().__init__(key="tpot", lower_is_better=True)


class _TbtMetricSpec(_BaseMetricSpec):
    def __init__(self) -> None:
        super().__init__(key="tbt", lower_is_better=True)

    def extract(self, request_metrics: Dict[str, Any]) -> List[float]:  # type: ignore[override]
        values = request_metrics.get(self.key, [])
        if not values:
            return []
        # request-level structure is List[List[float]]; flatten for percentile eval
        if isinstance(values, list) and values and isinstance(values[0], list):
            return [item for sublist in values for item in sublist]
        return values if isinstance(values, list) else []


class _DeadlineMissRateMetricSpec(_BaseMetricSpec):
    def __init__(self) -> None:
        super().__init__(key="deadline_miss_rate", lower_is_better=True)


class MetricRegistry:
    _registry: Dict[str, MetricSpec] = {}

    @classmethod
    def register(cls, spec: MetricSpec) -> None:
        cls._registry[spec.key] = spec

    @classmethod
    def get(cls, key: str) -> MetricSpec:
        if key not in cls._registry:
            raise ValueError(f"Metric '{key}' is not registered")
        return cls._registry[key]

    @classmethod
    def has(cls, key: str) -> bool:
        return key in cls._registry


# Register built-in metrics
MetricRegistry.register(_TtftMetricSpec())
MetricRegistry.register(_TbtMetricSpec())
MetricRegistry.register(_TpotMetricSpec())
MetricRegistry.register(_DeadlineMissRateMetricSpec())
