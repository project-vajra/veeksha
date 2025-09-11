import os
from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.interval_generator.base_generator import (
    BaseRequestIntervalGeneratorConfig,
)
from veeksha.constants.configuration_constants import ALLOWED_TS_UNITS
from veeksha.types.request_interval_generator_type import RequestIntervalGeneratorType


@frozen_dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    exhaustion_policy: str = field(
        default="stop",
        metadata={
            "help": "Behavior when the trace runs out: error | stop | wrap.",
        },
    )
    trace_file: str = field(
        default="data/processed_traces/swe_agent_trace_short.jsonl",
        metadata={
            "help": "Path to the trace file for request intervals. Should be a csv or jsonl file."
        },
    )
    timestamp_column: str = field(
        default="timestamp",
        metadata={"help": "Name of the column containing request timestamps."},
    )
    timestamp_unit: str = field(
        default="ms",
        metadata={
            "help": f"Unit of the timestamps in the trace file. Must be in {sorted(ALLOWED_TS_UNITS)}."
        },
    )
    time_scale_factor: float = field(
        default=1,
        metadata={"help": "Factor to scale the dispatch intervals in the trace."},
    )

    def __post_init__(self):
        # check if trace file exists
        if not os.path.exists(self.trace_file):
            raise FileNotFoundError(
                f"{self.__class__.__name__}: Trace file not found: {self.trace_file}"
            )
        # time_scale_factor cannot be negative
        if self.time_scale_factor < 0:
            raise ValueError(
                f"{self.__class__.__name__}: time_scale_factor cannot be negative"
            )
        if self.timestamp_unit not in ALLOWED_TS_UNITS:
            raise ValueError(
                f"{self.__class__.__name__}: timestamp_unit must be in {sorted(ALLOWED_TS_UNITS)}"
            )
        if self.exhaustion_policy not in {"error", "stop", "wrap"}:
            raise ValueError(
                f"{self.__class__.__name__}: exhaustion_policy must be one of ['error','stop','wrap']"
            )

    @classmethod
    def get_type(cls):
        return RequestIntervalGeneratorType.TRACE
