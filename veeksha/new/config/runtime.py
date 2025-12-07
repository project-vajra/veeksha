from dataclasses import field

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class RuntimeConfig:
    num_prefetch_threads: int = field(
        default=4,
        metadata={"help": "Number of threads for prefetching/generating requests."},
    )
    num_dispatcher_threads: int = field(
        default=4,
        metadata={"help": "Number of threads for dispatching requests to workers."},
    )
    num_results_processor_threads: int = field(
        default=4,
        metadata={"help": "Number of threads for processing completed requests."},
    )
    num_request_runner_threads: int = field(
        default=10,
        metadata={
            "help": "Number of async worker threads for making concurrent requests."
        },
    )
    telemetry_enabled: bool = field(
        default=False,
        metadata={
            "help": "Enable verbose dispatch runtime telemetry logs (backlog, prefetch rate)."
        },
    )
