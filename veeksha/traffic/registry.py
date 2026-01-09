from veeksha.core.lazy_loader import _LazyLoader
from veeksha.types import TrafficType
from veeksha.types.base_registry import BaseRegistry


class TrafficSchedulerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> TrafficType:
        return TrafficType.from_str(key_str)  # type: ignore


TrafficSchedulerRegistry.register(
    TrafficType.RATE,
    _LazyLoader(
        "veeksha.traffic.rate",
        "RateTrafficScheduler",
    ),
)

TrafficSchedulerRegistry.register(
    TrafficType.CONCURRENT,
    _LazyLoader(
        "veeksha.traffic.concurrent",
        "ConcurrentTrafficScheduler",
    ),
)
