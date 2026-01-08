"""Thread pool manager for worker threads."""

import threading
from threading import Thread
from typing import List

from veeksha.core.context import WorkerContext
from veeksha.logger import init_logger

logger = init_logger(__name__)


class ThreadPoolManager:
    """Manages creation and lifecycle of worker thread pools."""

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.thread_pools: dict[str, List[Thread]] = {}

    def create_pool(
        self, name: str, worker_class, worker_kwargs: dict, pool_size: int
    ) -> List[Thread]:
        """Create a pool of worker threads.

        Args:
            name: Name prefix for the threads
            worker_class: Worker class to instantiate
            worker_kwargs: Common kwargs for all workers (worker_context will be added)
            pool_size: Number of threads in the pool

        Returns:
            List of created threads
        """
        threads = []
        for i in range(pool_size):
            # Create worker context for this worker
            worker_context = WorkerContext(worker_id=i, stop_event=self.stop_event)

            # Create worker instance with worker_context
            worker = worker_class(**worker_kwargs, worker_context=worker_context)

            # Create and start thread
            thread = Thread(
                target=worker.run,
                name=f"{name}-{i}",
                daemon=False,
            )
            threads.append(thread)

        self.thread_pools[name] = threads
        return threads

    def start_all(self) -> None:
        """Start all threads in all pools."""
        for threads in self.thread_pools.values():
            for thread in threads:
                thread.start()

    def join_pool(self, name: str, timeout: float) -> None:
        """Wait for all threads in a specific pool to complete."""
        if name in self.thread_pools:
            for thread in self.thread_pools[name]:
                thread.join(timeout=timeout)
            logger.debug(f"All {len(self.thread_pools[name])} {name} threads joined")

    def join_all(self, timeout: float) -> None:
        """Wait for all threads in all pools to complete."""
        for name in self.thread_pools.keys():
            self.join_pool(name, timeout=timeout)

    def get_total_thread_count(self) -> int:
        """Get total number of threads across all pools."""
        return sum(len(threads) for threads in self.thread_pools.values())
