from typing import Tuple

from veeksha.config.generators.length_generator.trace_generator import (
    TraceRequestLengthGeneratorConfig,
)
from veeksha.generators.length_generator.base_generator import (
    BaseRequestLengthGenerator,
)
from veeksha.generators.utils import load_trace, process_request_length_trace
from veeksha.logger import init_logger

logger = init_logger(__name__)


class TraceRequestLengthGenerator(BaseRequestLengthGenerator):
    def __init__(self, config: TraceRequestLengthGeneratorConfig):
        self.config = config

        raw_trace_df = load_trace(self.config.trace_file)

        # canonical column names
        self.length_column_map = {
            self.config.input_length_column: "input_length",
            self.config.output_length_column: "output_length",
        }

        self.trace_df = process_request_length_trace(
            raw_trace_df,
            self.config.trace_file,
            self.length_column_map,
            self.config.prefill_scale_factor,
            self.config.decode_scale_factor,
            self.config.max_tokens,
        )

        logger.info(
            f"Loaded request length trace file {self.config.trace_file} with {len(self.trace_df)} requests"
        )

        self.next_request_idx = 0
        self._wrap_warning_logged = False

    def get_next_num_tokens(self) -> Tuple[int, int]:
        if self.next_request_idx >= self.capacity():
            if self.config.exhaustion_policy == "error":
                raise StopIteration(
                    f"Trace exhausted for lengths at index {self.next_request_idx}"
                )
            elif self.config.exhaustion_policy == "stop":
                logger.info(
                    f"Stop policy active: length trace exhausted at index {self.next_request_idx}."
                )
                return -1, -1
            elif self.config.exhaustion_policy == "wrap":
                if not self._wrap_warning_logged:
                    logger.warning(
                        f"Length trace exhausted at index {self.next_request_idx}; wrapping to start."
                    )
                    self._wrap_warning_logged = True
                self.next_request_idx = 0

        row = self.trace_df.iloc[self.next_request_idx]
        self.next_request_idx += 1

        return (
            int(row["input_length"]),
            int(row["output_length"]),
        )

    def capacity(self) -> int:
        return len(self.trace_df)
