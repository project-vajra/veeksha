from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.interval_generator.base_generator import (
    BaseRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.interval_generator.poisson_generator import (
    PoissonRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.length_generator.base_generator import (
    BaseRequestLengthGeneratorConfig,
)
from veeksha.config.generators.length_generator.trace_generator import (
    TraceRequestLengthGeneratorConfig,
)
from veeksha.config.generators.request_generator.base_generator import (
    BaseRequestGeneratorConfig,
)
from veeksha.types.request_generator_type import RequestGeneratorType


@frozen_dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=TraceRequestLengthGeneratorConfig
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig
    )
    save_to_trace: bool = field(
        default=False,
        metadata={
            "help": "If True, save generated requests to a trace file for future use."
        },
    )
    trace_file_path: str = field(
        default="synthetic_requests_trace.jsonl",
        metadata={"help": "Path to save the trace file when save_to_trace is True."},
    )

    @classmethod
    def get_type(cls):
        return RequestGeneratorType.SYNTHETIC
