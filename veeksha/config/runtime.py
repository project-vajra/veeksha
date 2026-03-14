from vidhi import field, frozen_dataclass


@frozen_dataclass
class WarmupRequestConfig:
    enabled: bool = field(
        False,
        help="Whether to send a small synthetic request to the server before the benchmark starts.",
    )
    prompt: str = field(
        "Warm up.",
        help="Prompt text for the pre-benchmark warmup request.",
    )
    output_tokens: int = field(
        8,
        help="Target number of output tokens for the pre-benchmark warmup request.",
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
    warmup_request: WarmupRequestConfig = field(
        default_factory=WarmupRequestConfig,
        help="Optional single request sent after server startup and before benchmark timing begins.",
    )
