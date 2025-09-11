from dataclasses import field
from typing import List

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.logger import init_logger

logger = init_logger(__name__)


@frozen_dataclass(allow_from_file=True)
class PrefillProfilerConfig:
    prefill_lengths: List[int] = field(
        default_factory=lambda: [2**i for i in range(8, 15)],
        metadata={"help": "The lengths to prefill the profiler with."},
    )
