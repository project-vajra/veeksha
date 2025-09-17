import os
from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.interval_generator.base_generator_config import (
    BaseRequestIntervalGeneratorConfig,
)
from veeksha.config.utils import get_trace_file_path
from veeksha.constants.configuration_constants import (
    ALLOWED_EXHAUSTION_POLICIES,
    ALLOWED_TS_UNITS,
)
from veeksha.types.request_interval_generator_type import RequestIntervalGeneratorType

_DATA_FILE_PATH = get_trace_file_path("swe_agent_trace_short.jsonl")
DEFAULT_TRACE_FILE = str(_DATA_FILE_PATH)


@frozen_dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    exhaustion_policy: str = field(
        default="stop",
        metadata={
            "help": "Behavior when the trace runs out: error | stop | wrap.",
        },
    )
    trace_file: str = field(
        default=DEFAULT_TRACE_FILE,
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
        assert os.path.exists(self.trace_file), f"Trace interval generator: Trace file not found: {self.trace_file}"
        assert self.time_scale_factor >= 0, f"Trace interval generator: time_scale_factor cannot be negative"
        assert self.timestamp_unit in ALLOWED_TS_UNITS, f"Trace interval generator: timestamp_unit must be in {sorted(ALLOWED_TS_UNITS)}"   
        assert self.exhaustion_policy in ALLOWED_EXHAUSTION_POLICIES, f"Trace interval generator: exhaustion_policy must be one of {sorted(ALLOWED_EXHAUSTION_POLICIES)}"

    @classmethod
    def get_type(cls):
        return RequestIntervalGeneratorType.TRACE
