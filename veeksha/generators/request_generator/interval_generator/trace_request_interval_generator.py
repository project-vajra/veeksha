from veeksha.config.generators.interval_generator.trace_generator_config import (
    TraceRequestIntervalGeneratorConfig,
)
from veeksha.generators.request_generator.interval_generator.base_request_interval_generator import (
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
        self._wrap_warning_logged = False

    def get_next_inter_request_time(self) -> float:
        if self.next_request_idx >= self.capacity():
            if self.config.exhaustion_policy == "error":
                raise StopIteration(
                    f"Trace exhausted for intervals at index {self.next_request_idx}"
                )
            elif self.config.exhaustion_policy == "stop":
                logger.info(
                    f"Stop policy active: interval trace exhausted at index {self.next_request_idx}."
                )
                return -1
            elif self.config.exhaustion_policy == "wrap":
                if not self._wrap_warning_logged:
                    logger.warning(
                        f"Interval trace exhausted at index {self.next_request_idx}; wrapping to start."
                    )
                    self._wrap_warning_logged = True
                self.next_request_idx = 0

        inter_request_time = self.trace_df.iloc[self.next_request_idx][
            "inter_request_time"
        ]
        self.next_request_idx += 1
        return inter_request_time

    def capacity(self) -> int:
        return len(self.trace_df)
