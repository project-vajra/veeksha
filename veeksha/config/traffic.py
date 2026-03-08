from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generator.interval import (
    BaseIntervalGeneratorConfig,
    PoissonIntervalGeneratorConfig,
)
from veeksha.types import TrafficType


@frozen_dataclass(allow_from_file=True)
class BaseTrafficConfig(BasePolyConfig):
    cancel_session_on_failure: bool = field(
        default=True,
        metadata={"help": "Whether to cancel the session on failure of any request."},
    )


@frozen_dataclass
class RateTrafficConfig(BaseTrafficConfig):
    interval_generator: BaseIntervalGeneratorConfig = field(
        default_factory=PoissonIntervalGeneratorConfig,
        metadata={
            "help": "Interval generator for the traffic (sessions per second). Available: poisson, gamma, fixed."
        },
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
    rampup_seconds: int = field(
        default=10,
        metadata={
            "help": "Number of seconds to ramp up the traffic. i.e. 'Take 10 seconds to ramp up to the target concurrent sessions.'"
        },
    )

    @classmethod
    def get_type(cls) -> TrafficType:
        return TrafficType.CONCURRENT


@frozen_dataclass
class SequentialLaunchTrafficConfig(BaseTrafficConfig):
    """Launch sessions one at a time: the next session is activated only after
    the previous session's request has been dispatched.

    All sessions remain concurrently active on the server — only the *launch
    order* is sequential.  This guarantees engine-level FCFS ordering because
    requests arrive at the server one at a time.
    """

    ordering: str = field(
        default="dispatch",
        metadata={
            "help": (
                "When to advance the dispatch ticket and unblock the next request. "
                "dispatch: after HTTP 200 (before streaming); "
                "prefill: after first content chunk (TTFC); "
                "request: after full completion."
            ),
        },
    )

    def __post_init__(self) -> None:
        allowed = ("dispatch", "prefill", "request")
        if self.ordering not in allowed:
            raise ValueError(
                f"Invalid ordering {self.ordering!r}; must be one of {allowed}"
            )

    @classmethod
    def get_type(cls) -> TrafficType:
        return TrafficType.SEQUENTIAL_LAUNCH
