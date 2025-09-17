from dataclasses import field
from typing import List

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.microbenchmark import BaseMicrobenchmarkProbeConfig
from veeksha.types.microbenchmark_probe_type import MicrobenchmarkProbeType


@frozen_dataclass
class PrefillProbeConfig(BaseMicrobenchmarkProbeConfig):
    prefill_lengths: List[int] = field(
        default_factory=lambda: [2**i for i in range(8, 15)],
        metadata={"help": "The prompt lengths to probe prefill time with."},
    )
    num_requests_per_prefill_length: int = field(
        default=1,
        metadata={
            "help": "Number of completed requests per prompt length before stopping.",
        },
    )

    @classmethod
    def get_type(cls):
        return MicrobenchmarkProbeType.PREFILL
