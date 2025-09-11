from veeksha.config.generators.interval_generator.trace_generator import (
    TraceRequestIntervalGeneratorConfig,
)
from veeksha.generators.interval_generator.base_generator import (
    BaseRequestIntervalGenerator,
)
from veeksha.generators.utils import load_trace, process_request_interval_trace
from veeksha.logger import init_logger

logger = init_logger(__name__)


class TraceRequestIntervalGenerator(BaseRequestIntervalGenerator):
    """
    Reads a trace csv file containing request arrival time, its prompt and completion token values to generate
    inter-request times, number of tokens.
    """

    def __init__(self, config: TraceRequestIntervalGeneratorConfig):
        self.config = config

        raw_trace_df = load_trace(self.config.trace_file)

        # canonical column names
        self.interval_column_map = {self.config.timestamp_column: "timestamp"}

        self.trace_df = process_request_interval_trace(
            raw_trace_df,
            self.config.trace_file,
            self.interval_column_map,
            self.config.time_scale_factor,
            self.config.timestamp_unit,
        )

        logger.info(
            f"Loaded interval trace file {self.config.trace_file} with {len(self.trace_df)} requests"
        )

        self.next_request_idx = 0

    def get_next_inter_request_time(self) -> float:
        if self.next_request_idx >= len(self.trace_df):
            if self.config.exhaustion_policy == "error":
                raise StopIteration(
                    f"Trace exhausted for intervals at index {self.next_request_idx}"
                )
            return -1

        inter_request_time = self.trace_df.iloc[self.next_request_idx][
            "inter_request_time"
        ]
        self.next_request_idx += 1
        return inter_request_time

    def capacity(self):
        return len(self.trace_df)
