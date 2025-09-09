from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class DecodeProfilerConfig:
    context_lengths: list[int] = field(
        default_factory=lambda: [2**i for i in range(8, 15)],
        metadata={"help": "The lengths to decode the profiler with."},
    )
    engine_chunk_size: int = field(
        default=512,
        metadata={"help": "The chunk size the engine is running with."},
    )
    batch_sizes: list[int] = field(
        default_factory=lambda: [2**i for i in range(4, 8)],
        metadata={"help": "The batch sizes to decode the profiler with."},
    )
    enable_mixed_batching: bool = field(
        default=False,
        metadata={"help": "Whether to enable mixed batching."},
    )