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
        
        trace_df = load_trace(self.config.trace_file)

        self.trace_df = process_request_length_trace(
            trace_df,
            self.config.trace_file,
            self.config.prefill_scale_factor,
            self.config.decode_scale_factor,
            self.config.max_tokens,
        )

        logger.info(
            f"Loaded request length trace file {self.config.trace_file} with {len(self.trace_df)} requests"
        )

        self.next_request_idx = 0

    def get_next_num_tokens(self) -> Tuple[int, int]:
        if self.next_request_idx >= len(self.trace_df):
            return -1, -1

        row = self.trace_df.iloc[self.next_request_idx]
        self.next_request_idx += 1

        return (
            int(row["input_length"]),
            int(row["output_length"]),
        )
