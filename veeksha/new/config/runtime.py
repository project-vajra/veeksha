from dataclasses import field

from veeksha.new.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class RuntimeConfig:
    max_sessions: int = field(
        default=-1,
        metadata={"help": "Maximum number of sessions to generate. -1 for unlimited."},
    )
    num_prefetch_threads: int = field(
        default=2,
        metadata={"help": "Number of threads for prefetching/generating sessions."},
    )
    num_dispatcher_threads: int = field(
        default=2,
        metadata={"help": "Number of threads for dispatching requests to workers."},
    )
    num_completion_threads: int = field(
        default=2,
        metadata={"help": "Number of threads for processing completed requests."},
    )
    num_client_threads: int = field(
        default=4,
        metadata={
            "help": "Number of async worker threads for making concurrent requests."
        },
    )
    timeout: int = field(
        default=300,
        metadata={"help": "Request timeout in seconds."},
    )
    # TODO rm
    telemetry_enabled: bool = field(
        default=False,
        metadata={"help": "Enable verbose dispatch runtime telemetry logs."},
    )
