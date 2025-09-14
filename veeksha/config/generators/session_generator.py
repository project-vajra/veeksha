from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.interval_generator.base_generator import (
    BaseRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.interval_generator.poisson_generator import (
    PoissonRequestIntervalGeneratorConfig,
)
from veeksha.logger import init_logger

logger = init_logger(__name__)


@frozen_dataclass(allow_from_file=True)  # either dataclass or frozen dataclass
class SessionGeneratorConfig:
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    session_interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig,
        metadata={
            "help": "Interval generator for session dispatch. This will determine how often sessions are dispatched."
        },
    )
    minimum_prefix_match: float = field(
        default=0.8,
        metadata={
            "help": "Minimum pct. of prefix match between requests in a session."
        },
    )
    min_session_size: int = field(
        default=1,
        metadata={"help": "Minimum number of requests per session."},
    )
    max_session_size: int = field(
        default=10,
        metadata={"help": "Maximum number of requests per session."},
    )
    max_request_interval: float = field(
        default=600.0,
        metadata={
            "help": "Maximum time interval between consecutive requests in a session, in seconds."
        },
    )
    save_as_trace_file: bool = field(
        default=False,
        metadata={
            "help": "If true, save the trace after session generation as a jsonl file."
        },
    )
    trace_file_save_dir: str = field(
        default="./generated_traces",
        metadata={
            "help": "Directory to save generated trace files when save_as_trace_file is true."
        },
    )
    trace_file_name: str = field(
        default="",
        metadata={
            "help": "If save_as_trace_file is true, this is the name of the trace file, without the extension. Config params will be appended to the file name."
        },
    )

    def __post_init__(self):
        if self.trace_file_name != "" and not self.save_as_trace_file:
            logger.warning(
                "trace_file_name is provided but save_as_trace_file is false. trace_file_name will be ignored."
            )
