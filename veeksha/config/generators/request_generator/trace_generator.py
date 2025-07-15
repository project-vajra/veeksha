import os
from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.request_generator.base_generator import (
    BaseRequestGeneratorConfig,
)
from veeksha.config.generators.session_generator import (
    SessionGeneratorConfig,
)
from veeksha.types.request_generator_type import RequestGeneratorType


@frozen_dataclass(allow_from_file=True)
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/swe_agent_trace_short.jsonl",
        metadata={"help": "Path to the trace file for request generation."},
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
    time_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for request dispatch intervals."}
    )
    use_trace_prefix_hash_ids: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, veeksha will use prefix hash IDs of requests to generate request prompts. Trace file specified by interval or/and length generator must include hash_ids: list[int]."
        },
    )
    use_trace_sessions: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, veeksha will use sessions provided in the trace file (session_id: int)."
        },
    )
    session_generator_config: Optional[SessionGeneratorConfig] = field(
        default=None,
        metadata={
            "help": "If not None, it will synthesize sessions based on the trace file and prefix hash IDs of requests (requires use_prefix_hash_ids to be True)."
        },
    )

    def __post_init__(self):
        # check if trace file exists
        if not os.path.exists(self.trace_file):
            raise FileNotFoundError(
                f"{self.__class__.__name__}: Trace file not found: {self.trace_file}"
            )
        # factors cannot be negative
        if self.prefill_scale_factor < 0:
            raise ValueError(
                f"{self.__class__.__name__}: prefill_scale_factor cannot be negative"
            )
        if self.decode_scale_factor < 0:
            raise ValueError(
                f"{self.__class__.__name__}: decode_scale_factor cannot be negative"
            )
        if self.time_scale_factor < 0:
            raise ValueError(
                f"{self.__class__.__name__}: time_scale_factor cannot be negative"
            )

        # block_size must be > 0
        if self.block_size <= 0:
            raise ValueError(f"{self.__class__.__name__}: block_size must be positive")

        # session_generator_config and use_trace_sessions cannot both be provided
        if self.session_generator_config and self.use_trace_sessions:
            raise ValueError(
                f"{self.__class__.__name__}: session_generator_config and use_trace_sessions cannot both be provided"
            )
        # if session_generator_config is provided, use_trace_prefix_hash_ids must be True
        if self.session_generator_config and not self.use_trace_prefix_hash_ids:
            raise ValueError(
                f"{self.__class__.__name__}: session_generator_config requires use_trace_prefix_hash_ids to be True"
            )

    @classmethod
    def get_type(cls):
        return RequestGeneratorType.TRACE
