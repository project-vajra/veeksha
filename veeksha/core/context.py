"""Context objects for benchmark runtime."""

import threading
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class WorkerContext:
    """Context object for worker threads."""

    worker_id: int
    stop_event: threading.Event
    # Used for power-of-two load balancing
    _running_tasks: int = field(default=0, init=False)
    _load_lock: Lock = field(default_factory=Lock, init=False)

    def increment_load(self) -> None:
        """Increment the number of running tasks."""
        with self._load_lock:
            self._running_tasks += 1

    def decrement_load(self) -> None:
        """Decrement the number of running tasks."""
        with self._load_lock:
            self._running_tasks -= 1

    def get_load(self) -> int:
        """Get the current number of running tasks."""
        with self._load_lock:
            return self._running_tasks
