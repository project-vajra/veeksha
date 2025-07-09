from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.request_generator.base_generator import (
    BaseRequestGeneratorConfig,
)
from veeksha.config.generators.synthetic_session_generator import (
    SyntheticSessionGeneratorConfig,
)
from veeksha.types.request_generator_type import RequestGeneratorType


@frozen_dataclass
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
            "help": "If True, veeksha will use prefix hash IDs of requests to generate request inputs. Trace file specified by interval or/and length generator must include hash_ids: list[int]."
        },
    )
    use_trace_sessions: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, veeksha will use sessions provided in the trace file (session_id: int)."
        },
    )
    session_generator_config: Optional[SyntheticSessionGeneratorConfig] = field(
        default=None,
        metadata={
            "help": "If not None, it will synthesize sessions based on the trace file and prefix hash IDs of requests (requires use_prefix_hash_ids to be True)."
        },
    )

    @classmethod
    def get_type(cls):
        return RequestGeneratorType.TRACE
