"""Process-wide executor for blocking work offloaded from async event loops.

``loop.run_in_executor(None, ...)`` lazily creates a *per event loop* default
``ThreadPoolExecutor`` sized ``min(32, cpu_count + 4)``.  With one event loop
per client worker thread that is up to ``num_client_threads * 32`` threads
nobody asked for, so the process oversubscribes the machine by an amount that
depends on load rather than on configuration.

This module replaces those anonymous pools with a single explicitly sized one:
the benchmark reserves a thread per named worker (client, dispatch, completion,
prefetch) and this executor gets whatever is left of the CPU budget, so the
total thread count is deterministic.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from veeksha.logger import init_logger

logger = init_logger(__name__)

_executor: Optional[ThreadPoolExecutor] = None
_lock = threading.Lock()


def compute_blocking_thread_count(
    num_client_threads: int,
    num_dispatcher_threads: int,
    num_completion_threads: int,
    num_prefetch_threads: int,
) -> int:
    """Return the number of threads left over for blocking work.

    The target is one thread per core (``os.process_cpu_count()`` honors CPU
    affinity, so a cgroup/taskset-pinned run gets its own budget), minus the
    named worker threads the benchmark starts itself.
    """
    total_cores = os.process_cpu_count() or os.cpu_count() or 1
    reserved = (
        num_client_threads
        + num_dispatcher_threads
        + num_completion_threads
        + num_prefetch_threads
    )
    budget = total_cores - reserved

    # Only two call sites offload onto this executor, and both belong to a
    # client worker: the permanently parked ``input_queue.get`` (one slot per
    # worker, held for the whole run) and ``STTClient._clip_assets`` (bounded,
    # returns on its own).  So ``num_client_threads`` slots are permanently
    # unavailable, and a second ``num_client_threads`` guarantees every worker
    # a free slot for a clip decode no matter what its peers are doing --
    # ``2 * num_client_threads`` is the deterministic no-deadlock floor.  It
    # wins over the CPU budget.
    #
    # Ordering waits are not counted: ``DispatchTracker.wait_for_turn`` is a
    # coroutine suspension and consumes no slot.
    deadlock_free_floor = num_client_threads * 2

    if budget < deadlock_free_floor:
        logger.warning(
            "CPU budget leaves %d thread(s) for blocking work (%d cores - %d "
            "worker threads); raising to %d so client workers do not starve. "
            "Consider lowering the worker thread counts.",
            budget,
            total_cores,
            reserved,
            deadlock_free_floor,
        )
        return deadlock_free_floor
    return budget


def start_blocking_executor(num_threads: int) -> ThreadPoolExecutor:
    """Create the process-wide blocking executor.

    Idempotent per benchmark run: an already-running executor is returned as
    is, so a nested/second run does not silently orphan the first pool.
    """
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=num_threads,
                thread_name_prefix="blocking",
            )
            logger.info("Started blocking executor with %d threads", num_threads)
        return _executor


def get_blocking_executor() -> Optional[ThreadPoolExecutor]:
    """Return the shared executor, or ``None`` when it is not running.

    ``None`` is exactly what ``run_in_executor`` wants as a fallback, so call
    sites work unchanged outside a benchmark run (unit tests, warmup).
    """
    return _executor


def shutdown_blocking_executor() -> None:
    """Shut down the shared executor without waiting for parked threads.

    Client workers leave threads blocked on a closed queue's ``get`` during
    teardown; ``wait=True`` would hang on them.
    """
    global _executor
    with _lock:
        executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
        logger.debug("Blocking executor shut down")
