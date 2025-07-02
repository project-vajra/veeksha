from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.length_generator.base_generator import (
    BaseRequestLengthGeneratorConfig,
)
from veeksha.types import RequestLengthGeneratorType


@frozen_dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
        metadata={"help": "Path to the trace file for request lengths."},
    )
    max_tokens: int = field(
        default=-1,
        metadata={
            "help": "Maximum number of tokens allowed in a request. -1 for following the trace file."
        },
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

    @classmethod
    def get_type(cls) -> RequestLengthGeneratorType:
        return RequestLengthGeneratorType.TRACE
