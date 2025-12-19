from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.new.config.core.frozen_dataclass import frozen_dataclass
from veeksha.new.config.generator.interval import (
    BaseIntervalGeneratorConfig,
    PoissonIntervalGeneratorConfig,
)
from veeksha.new.types import TrafficType


@frozen_dataclass
class BaseTrafficConfig(BasePolyConfig):
    cancel_session_on_failure: bool = field(
        default=True,
        metadata={"help": "Whether to cancel the session on failure of any request."},
    )


@frozen_dataclass
class RateTrafficConfig(BaseTrafficConfig):
    interval_generator: BaseIntervalGeneratorConfig = field(
        default_factory=PoissonIntervalGeneratorConfig,
        metadata={"help": "Interval generator for the traffic (sessions per second)."},
    )

    @classmethod
    def get_type(cls) -> TrafficType:
        return TrafficType.RATE


@frozen_dataclass
class ConcurrentTrafficConfig(BaseTrafficConfig):
    target_concurrent_sessions: int = field(
        default=3,
        metadata={"help": "Target number of concurrent sessions to maintain."},
    )

    @classmethod
    def get_type(cls) -> TrafficType:
        return TrafficType.CONCURRENT
