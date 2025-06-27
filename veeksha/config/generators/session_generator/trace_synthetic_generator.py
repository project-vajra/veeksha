from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.interval_generator.base_generator import BaseRequestIntervalGeneratorConfig
from veeksha.config.generators.interval_generator.poisson_generator import PoissonRequestIntervalGeneratorConfig
from veeksha.config.generators.session_generator.base_generator import BaseSessionGeneratorConfig
from veeksha.types.session_generator_type import SessionGeneratorType


@frozen_dataclass
class TraceSyntheticSessionGeneratorConfig(BaseSessionGeneratorConfig):
    session_interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig,
        metadata={"help": "Interval generator for session dispatch. This will determine how often sessions are dispatched."}
    )
    minimum_prefix_match: float = field(
        default=0.8,
        metadata={"help": "Minimum pct. of prefix match between requests in a session."},
    )
    min_session_size: int = field(
        default=1,
        metadata={"help": "Minimum number of requests per session."},
    )
    max_session_size: int = field(
        default=10,
        metadata={"help": "Maximum number of requests per session."},
    )
    max_request_interval: float = field(
        default=1.0,
        metadata={"help": "Maximum time interval between consecutive requests in a session, in seconds."},
    )

    @classmethod
    def get_type(cls):
        return SessionGeneratorType.TRACE_SYNTHETIC
