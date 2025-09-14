from dataclasses import field
from typing import List

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.types.microbenchmark_probe_type import MicrobenchmarkProbeType
from veeksha.config.microbenchmark import BaseMicrobenchmarkProbeConfig

@frozen_dataclass
class DecodeProbeConfig(BaseMicrobenchmarkProbeConfig):
    context_lengths: List[int] = field(
        default_factory=lambda: [2**i for i in range(8, 15)],
        metadata={"help": "The context lengths to probe decode performance with."},
    )
    engine_chunk_size: int = field(
        default=512, metadata={"help": "The chunk size the engine is running with."}
    )
    batch_sizes: List[int] = field(
        default_factory=lambda: [2**i for i in range(4, 8)],
        metadata={"help": "The batch sizes to probe decode with."},
    )
    engine_uses_mixed_batching: bool = field(
        default=False, metadata={"help": "Whether the engine uses mixed batching."}
    )
    num_concurrent_requests_per_client: int = field(
        default=10,
        metadata={
            "help": "Concurrent requests per client used to size clients for decode probe.",
        },
    )
    profiling_iterations: int = field(
        default=100,
        metadata={
            "help": "Extra iterations to ensure enough decode tokens for batch overlap window.",
        },
    )

    @classmethod
    def get_type(cls):
        return MicrobenchmarkProbeType.DECODE