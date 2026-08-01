"""Tests for the shared blocking executor sizing and lifecycle."""

from unittest.mock import patch

from veeksha.core import blocking_executor
from veeksha.core.blocking_executor import (
    compute_blocking_thread_count,
    get_blocking_executor,
    shutdown_blocking_executor,
    start_blocking_executor,
)


def _compute(cores: int, **kwargs: int) -> int:
    defaults = dict(
        num_client_threads=8,
        num_dispatcher_threads=4,
        num_completion_threads=8,
        num_prefetch_threads=1,
    )
    defaults.update(kwargs)
    with patch("os.process_cpu_count", return_value=cores):
        return compute_blocking_thread_count(**defaults)  # type: ignore[arg-type]


def test_extra_threads_are_the_cores_left_over():
    assert _compute(64) == 64 - (8 + 4 + 8 + 1)


def test_floor_is_two_slots_per_client_worker():
    # 16 cores - 21 worker threads is negative; the floor takes over so the
    # client workers parked on input_queue.get cannot starve the pool.  Two
    # slots per worker: one parked on the get, one free for a clip decode.
    assert _compute(16) == 8 * 2
    assert _compute(16, num_client_threads=3) == 3 * 2


def test_start_is_idempotent_and_shutdown_clears():
    try:
        executor = start_blocking_executor(3)
        assert get_blocking_executor() is executor
        assert start_blocking_executor(9) is executor
        assert executor._max_workers == 3
    finally:
        shutdown_blocking_executor()
    assert get_blocking_executor() is None


def test_get_returns_none_outside_a_run():
    assert blocking_executor.get_blocking_executor() is None
