import os
from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.length_generator.base_generator import (
    BaseRequestLengthGeneratorConfig,
)
from veeksha.config.utils import get_trace_file_path
from veeksha.constants.configuration_constants import ALLOWED_EXHAUSTION_POLICIES
from veeksha.types import RequestLengthGeneratorType

_DATA_FILE_PATH = get_trace_file_path("sharegpt_8k_filtered_stats_llama2_tokenizer.csv")

DEFAULT_TRACE_FILE = str(_DATA_FILE_PATH)


@frozen_dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    exhaustion_policy: str = field(
        default="stop",
        metadata={
            "help": "Behavior when the trace runs out: error | stop | wrap.",
        },
    )
    trace_file: str = field(
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
        if self.trace_file == DEFAULT_TRACE_FILE:
            # For the default path, use the is_file() method on the importlib.resources object
            if not _DATA_FILE_PATH.is_file():
                raise FileNotFoundError(
                    f"{self.__class__.__name__}: Default trace file resource not found."
                )
        else:
            # For user-provided paths, use os.path.exists
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
        if self.exhaustion_policy not in ALLOWED_EXHAUSTION_POLICIES:
            raise ValueError(
                f"{self.__class__.__name__}: exhaustion_policy must be one of {sorted(ALLOWED_EXHAUSTION_POLICIES)}"
            )

    @classmethod
    def get_type(cls) -> RequestLengthGeneratorType:
        return RequestLengthGeneratorType.TRACE
