"""Unit tests for DispatchTracker."""

import asyncio
import contextlib
import threading
import time

from veeksha.traffic.dispatch_tracker import DispatchTracker


def wait_until(predicate, timeout_s=2.0, interval_s=0.005):
    """Wait until predicate returns True or timeout."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def waiter_count(tracker: DispatchTracker) -> int:
    with tracker._lock:
        return len(tracker._waiters)


class LoopThread:
    """Runs an event loop on its own OS thread, like a client worker."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = threading.Event()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(self._started.set)
        self.loop.run_forever()

    def __enter__(self):
        self._thread.start()
        assert self._started.wait(2.0)
        return self

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(2.0)
        self.loop.close()

    def submit(self, coro):
        """Schedule a coroutine on this loop, returning a concurrent Future."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


def test_ordering_is_exposed():
    assert DispatchTracker().ordering == "dispatch"
    assert DispatchTracker(ordering="prefill").ordering == "prefill"


def test_ticket_at_or_below_counter_does_not_suspend():
    tracker = DispatchTracker()

    # Ticket 0 is always immediately callable against a fresh counter.
    asyncio.run(asyncio.wait_for(tracker.wait_for_turn(0), timeout=1.0))

    tracker.advance(4)
    asyncio.run(asyncio.wait_for(tracker.wait_for_turn(5), timeout=1.0))

    assert waiter_count(tracker) == 0

def test_advance_from_another_thread_wakes_waiter():
    tracker = DispatchTracker()

    with LoopThread() as worker:
        future = worker.submit(tracker.wait_for_turn(1))
        assert wait_until(lambda: waiter_count(tracker) == 1)
        assert not future.done()

        # advance() runs on the main thread; the waiter is parked on the
        # worker's loop.
        tracker.advance(0)

        future.result(timeout=2.0)

    assert waiter_count(tracker) == 0


def test_advance_wakes_only_tickets_at_or_below_counter():
    tracker = DispatchTracker()

    with LoopThread() as worker:
        futures = {t: worker.submit(tracker.wait_for_turn(t)) for t in (1, 2, 3)}
        assert wait_until(lambda: waiter_count(tracker) == 3)

        tracker.advance(1)  # counter -> 2, releases tickets 1 and 2

        futures[1].result(timeout=2.0)
        futures[2].result(timeout=2.0)
        assert wait_until(lambda: waiter_count(tracker) == 1)
        assert not futures[3].done()

        tracker.advance(2)  # counter -> 3
        futures[3].result(timeout=2.0)

    assert waiter_count(tracker) == 0


def test_out_of_order_advance_does_not_move_counter_backwards():
    tracker = DispatchTracker()
    tracker.advance(10)
    tracker.advance(2)

    with tracker._lock:
        assert tracker._counter == 11

    asyncio.run(asyncio.wait_for(tracker.wait_for_turn(11), timeout=1.0))


def test_waiters_wake_across_multiple_loops():
    tracker = DispatchTracker()

    with LoopThread() as worker_a, LoopThread() as worker_b:
        future_a = worker_a.submit(tracker.wait_for_turn(1))
        future_b = worker_b.submit(tracker.wait_for_turn(2))
        assert wait_until(lambda: waiter_count(tracker) == 2)

        tracker.advance(1)
        future_a.result(timeout=2.0)
        future_b.result(timeout=2.0)

    assert waiter_count(tracker) == 0


def test_cancelled_waiter_does_not_block_later_tickets():
    tracker = DispatchTracker()

    async def cancel_a_waiter():
        task = asyncio.ensure_future(tracker.wait_for_turn(1))
        await asyncio.sleep(0)
        assert waiter_count(tracker) == 1
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # The cancelled entry is reclaimed lazily, and ticket 2 still wakes.
        follower = asyncio.ensure_future(tracker.wait_for_turn(2))
        await asyncio.sleep(0)
        tracker.advance(1)
        await asyncio.wait_for(follower, timeout=1.0)

    asyncio.run(cancel_a_waiter())
    assert waiter_count(tracker) == 0


def test_advance_survives_a_waiter_whose_loop_was_closed():
    """Teardown closes worker loops; stale waiters must not break advance()."""
    tracker = DispatchTracker()

    loop = asyncio.new_event_loop()

    async def register_then_abandon():
        task = asyncio.ensure_future(tracker.wait_for_turn(1))
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    try:
        loop.run_until_complete(register_then_abandon())
    finally:
        loop.close()

    assert waiter_count(tracker) == 1

    # Would raise RuntimeError("Event loop is closed") without the guard.
    tracker.advance(0)
    assert waiter_count(tracker) == 0

    # The tracker is still usable afterwards.
    with LoopThread() as worker:
        future = worker.submit(tracker.wait_for_turn(5))
        assert wait_until(lambda: waiter_count(tracker) == 1)
        tracker.advance(4)
        future.result(timeout=2.0)


def test_concurrent_advances_release_every_waiter():
    tracker = DispatchTracker()
    num_tickets = 64

    with LoopThread() as worker:
        futures = [worker.submit(tracker.wait_for_turn(t)) for t in range(num_tickets)]
        assert wait_until(lambda: waiter_count(tracker) == num_tickets - 1)

        # Hammer advance() from several threads at once, out of order.
        threads = [
            threading.Thread(target=tracker.advance, args=(t,))
            for t in reversed(range(num_tickets))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2.0)

        for future in futures:
            future.result(timeout=2.0)

    assert waiter_count(tracker) == 0
