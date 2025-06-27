from typing import Tuple

from veeksha.config.generators.length_generator.trace_generator import TraceRequestLengthGeneratorConfig
from veeksha.logger import init_logger
from veeksha.generators.length_generator.base_generator import (
    BaseRequestLengthGenerator,
)
from veeksha.generators.utils import process_request_length_trace, load_trace

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

    def get_next_num_tokens(self) -> Tuple[float, float]:
        if self.next_request_idx >= len(self.trace_df):
            return -1, -1

        row = self.trace_df.iloc[self.next_request_idx]
        self.next_request_idx += 1

        return (
            row["input_length"],
            row["output_length"],
        )

    def get_next_request_params(self) -> Tuple[int, int]:
        if self.next_request_idx >= len(self.trace_df):
            return -1, -1

        row = self.trace_df.iloc[self.next_request_idx]
        self.next_request_idx += 1

        input_length = row["input_length"]
        output_length = row["output_length"]
        
        return int(input_length), int(output_length)
