from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass

from veeksha.config.generators.request_generator.base_generator import BaseRequestGeneratorConfig
from veeksha.types.request_generator_type import RequestGeneratorType
from veeksha.config.generators.session_generator.base_generator import BaseSessionGeneratorConfig


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
    use_prefix_hash_ids: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, veeksha will use prefix hash IDs of requests to generate request inputs. Trace file specified by interval or/and length generator must include hash_ids: list[int]."}
    )
    session_generator_config: Optional[BaseSessionGeneratorConfig] = field(
        default=None,
        metadata={"help": "If not None, it will determine how sessions are created. (SyntheticSessionGeneratorConfig requires use_prefix_hash_ids to be True to determine similarity between requests in a session)."}
    )

    @classmethod
    def get_type(cls):
        return RequestGeneratorType.TRACE