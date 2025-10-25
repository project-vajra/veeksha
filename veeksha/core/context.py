"""Context objects for benchmark runtime."""

import threading
from dataclasses import dataclass


@dataclass
class BenchmarkContext:
    """Context object containing benchmark-level information."""

    benchmark_id: str
    telemetry_enabled: bool


@dataclass
class WorkerContext:
    """Context object for worker threads."""

    worker_id: int
    stop_event: threading.Event
