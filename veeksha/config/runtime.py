from vidhi import field, frozen_dataclass


@frozen_dataclass
class ProfilingConfig:
    command: str = field(
        "", help="Shell command to execute for profiling (e.g., curl to trigger nsys)"
    )
    trigger: str = field(
        "all_decoding",
        help="Trigger mode: 'all_decoding', 'elapsed', 'any_in_flight'",
    )
    trigger_value: int = field(
        0,
        help="Threshold: in-flight count for 'all_in_flight', seconds for 'elapsed'",
    )


@frozen_dataclass
class RuntimeConfig:
    max_sessions: int = field(
        25, help="Maximum number of sessions to generate. -1 for unlimited."
    )
    benchmark_timeout: int = field(300, help="Benchmark timeout in seconds.")
    post_timeout_grace_seconds: int = field(
        -1,
        help="Grace period for in-flight requests after timeout. -1 waits for all, 0 exits immediately.",
    )
    num_dispatcher_threads: int = field(
        2, help="Number of threads for dispatching requests to workers."
    )
    num_completion_threads: int = field(
        2, help="Number of threads for processing completed requests."
    )
    num_client_threads: int = field(
        3, help="Number of async worker threads for making concurrent requests."
    )
    pregenerate_sessions: bool = field(
        False,
        help="Pre-generate all sessions before starting benchmark timer. "
        "Requires max_sessions > 0.",
    )
    profiling: ProfilingConfig = field(
        default_factory=ProfilingConfig, help="Profiling command config"
    )
