from dataclasses import field
from typing import List

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class DecodeProfilerConfig:
    context_lengths: List[int] = field(
        default_factory=lambda: [2**i for i in range(8, 15)],
        metadata={"help": "The lengths to decode the profiler with."},
    )
    engine_chunk_size: int = field(
        default=512,
        metadata={"help": "The chunk size the engine is running with."},
    )
    batch_sizes: List[int] = field(
        default_factory=lambda: [2**i for i in range(4, 8)],
        metadata={"help": "The batch sizes to decode the profiler with."},
    )
    engine_uses_mixed_batching: bool = field(
        default=False,
        metadata={"help": "Whether the engine uses mixed batching."},
    )