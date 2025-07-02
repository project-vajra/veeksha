from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.interval_generator.base_generator import (
    BaseRequestIntervalGeneratorConfig,
)
from veeksha.types.request_interval_generator_type import RequestIntervalGeneratorType


@frozen_dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv",
        metadata={
            "help": "Path to the trace file for request intervals. Should be a csv or jsonl file."
        },
    )
    time_scale_factor: float = field(
        default=0.3,
        metadata={"help": "Factor to scale the time intervals in the trace."},
    )

    @classmethod
    def get_type(cls):
        return RequestIntervalGeneratorType.TRACE
