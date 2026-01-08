from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class RuntimeConfig:
    max_sessions: int = field(
        default=25,
        metadata={"help": "Maximum number of sessions to generate. -1 for unlimited."},
    )
    benchmark_timeout: int = field(
        default=300,
        metadata={"help": "Benchmark timeout in seconds."},
    )
    post_timeout_grace_seconds: int = field(
        default=-1,
        metadata={
            "help": "Grace period for in-flight requests after timeout. -1 waits for all, 0 exits immediately."
        },
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
        default=3,
        metadata={
            "help": "Number of async worker threads for making concurrent requests."
        },
    )
