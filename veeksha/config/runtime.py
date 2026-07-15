from dataclasses import field
from typing import Optional

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
        default=8,
        metadata={
            "help": (
                "Number of threads for processing completed requests. "
                "Completion workers also run per-request ASR scoring "
                "(WER + interactivity alignment) concurrently, so "
                "under-provisioning stretches the post-run drain when many "
                "sessions finish together."
            )
        },
    )
    num_client_threads: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of async worker threads for making concurrent "
                "requests. None (default) provisions for the offered load: "
                "one thread per 8 target-concurrent sessions, floor 3 — "
                "under-provisioned client threads stretch realtime send "
                "cadence and invalidate high-concurrency latency metrics."
            )
        },
    )
    pregenerate_sessions: bool = field(
        default=False,
        metadata={
            "help": "Pre-generate all sessions before starting benchmark timer. "
            "Requires max_sessions > 0."
        },
    )
