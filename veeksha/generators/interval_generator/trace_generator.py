from veeksha.config.generators.interval_generator.trace_generator import TraceRequestIntervalGeneratorConfig
from veeksha.logger import init_logger
from veeksha.generators.interval_generator.base_generator import (
    BaseRequestIntervalGenerator,
)
from veeksha.generators.utils import process_request_interval_trace, load_trace

logger = init_logger(__name__)


class TraceRequestIntervalGenerator(BaseRequestIntervalGenerator):
    """
    Reads a trace csv file containing request arrival time, its prompt and completion token values to generate
    inter-request times, number of tokens.
    """

    def __init__(self, config: TraceRequestIntervalGeneratorConfig):
        self.config = config

        self.trace_df = load_trace(self.config.trace_file)

        self.trace_df = process_request_interval_trace(
            self.trace_df,
            self.config.trace_file,
            self.config.time_scale_factor,
        )

        logger.info(
            f"Loaded interval trace file {self.config.trace_file} with {len(self.trace_df)} requests"
        )

        self.next_request_idx = 0

    def get_next_inter_request_time(self) -> float:
        if self.next_request_idx >= len(self.trace_df):
            return -1

        inter_request_time = self.trace_df.iloc[self.next_request_idx][
            "inter_request_time"
        ]
        self.next_request_idx += 1
        return inter_request_time
