import os
from dataclasses import field
import importlib.resources

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.length_generator.base_generator import (
    BaseRequestLengthGeneratorConfig,
)
from veeksha.types import RequestLengthGeneratorType

_DATA_FILE_PATH = (
    importlib.resources.files("veeksha.data.processed_traces")
    .joinpath("sharegpt_8k_filtered_stats_llama2_tokenizer.csv")
)

DEFAULT_TRACE_FILE = str(_DATA_FILE_PATH)


@frozen_dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        # default="data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
        default=DEFAULT_TRACE_FILE,
        metadata={"help": "Path to the trace file for request lengths."},
    )
    input_length_column: str = field(
        default="num_prefill_tokens",
        metadata={"help": "Name of the column containing the input (prefill) length."},
    )
    output_length_column: str = field(
        default="num_decode_tokens",
        metadata={"help": "Name of the column containing the output (decode) length."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum number of tokens allowed in a request."},
    )
    prefill_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for prefill tokens."}
    )
    decode_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for decode tokens."}
    )
    block_size: int = field(
        default=512, metadata={"help": "Number of tokens per block."}
    )

    def __post_init__(self):
        # check if trace file exists
        if not os.path.exists(self.trace_file):
            raise FileNotFoundError(
                f"{self.__class__.__name__}: Trace file not found: {self.trace_file}"
            )

        # prefill_scale_factor and decode_scale_factor cannot be negative
        if self.prefill_scale_factor < 0:
            raise ValueError(
                f"{self.__class__.__name__}: prefill_scale_factor cannot be negative"
            )
        if self.decode_scale_factor < 0:
            raise ValueError(
                f"{self.__class__.__name__}: decode_scale_factor cannot be negative"
            )
        if self.input_length_column == self.output_length_column:
            raise ValueError(
                f"{self.__class__.__name__}: input_length_column and output_length_column must differ"
            )
        if not (self.max_tokens == -1 or self.max_tokens > 0):
            raise ValueError(
                f"{self.__class__.__name__}: max_tokens must be -1 (no cap) or > 0; got {self.max_tokens}"
            )
        # block_size must be > 0
        if self.block_size <= 0:
            raise ValueError(f"{self.__class__.__name__}: block_size must be positive")

    @classmethod
    def get_type(cls) -> RequestLengthGeneratorType:
        return RequestLengthGeneratorType.TRACE
