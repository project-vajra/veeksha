import asyncio
import heapq
import threading


class DispatchTracker:
    """Gates request dispatch using a ticket counter.

    Each request is assigned an incrementing ticket number.  A request may
    dispatch when ``ticket <= counter``.  After the HTTP request is sent,
    the counter advances so the next ticket unblocks.

    Waiters suspend on an :class:`asyncio.Future` rather than a condition
    variable, so an ordering wait costs no OS thread.  ``advance`` may be
    called from any thread; it wakes waiters on their own event loops via
    ``call_soon_threadsafe``.

    Ticket uniqueness
    -----------------
    Every queued ticket is unique.  Tickets are handed out by
    :class:`~veeksha.traffic.sequential_launch.SequentialLaunchTrafficScheduler`
    from a single counter incremented under its scheduler lock, so no two
    requests are ever assigned the same value.  Non-root requests keep the
    ``Request.dispatch_ticket`` default of ``0`` and therefore do share a
    ticket, but ``0`` is always ``<= counter`` (the counter starts at ``0``
    and only moves forward), so those take the fast path in
    :meth:`wait_for_turn` and never enter the waiter heap.

    The heap therefore orders on the ticket alone and needs no tiebreaker.
    """

    def __init__(self, ordering: str = "dispatch") -> None:
        self._counter = 0
        self._ordering = ordering

        # OS-level lock to protect shared state from multiple threads.
        # Critical for free-threaded (No-GIL) Python 3.14.
        self._lock = threading.Lock()

        # Min-heap of (ticket, future), ordered by ticket.  Queued tickets are
        # unique (see the class docstring), so the Futures — which are not
        # orderable — are never compared.
        self._waiters: list[tuple[int, asyncio.Future[None]]] = []

    @property
    def ordering(self) -> str:
        return self._ordering

    async def wait_for_turn(self, ticket: int) -> None:
        """Suspend until ``ticket <= counter``."""
        loop = asyncio.get_running_loop()

        with self._lock:
            if ticket <= self._counter:
                return

            fut = loop.create_future()
            heapq.heappush(self._waiters, (ticket, fut))

        # A cancelled waiter needs no cleanup here: ``advance`` reclaims it
        # once the counter reaches its ticket, and ``_resolve_future`` skips
        # futures that are already done.  The counter only moves forward, so
        # the leftover entry is bounded.
        await fut

    def advance(self, ticket: int) -> None:
        """Advance counter and wake waiters (thread-safe)."""
        ready = []
        with self._lock:
            self._counter = max(self._counter, ticket + 1)
            while self._waiters and self._waiters[0][0] <= self._counter:
                ready.append(heapq.heappop(self._waiters)[1])

        # Scheduling happens outside the lock: ``call_soon_threadsafe`` takes
        # the target loop's own lock and writes its wakeup pipe, which is not
        # work to hold a process-wide lock across.
        for fut in ready:
            try:
                fut.get_loop().call_soon_threadsafe(self._resolve_future, fut)
            except RuntimeError:
                # The waiter's loop was closed during teardown before its task
                # could run cancellation cleanup.  Nothing left to wake.
                pass

    @staticmethod
    def _resolve_future(fut: asyncio.Future[None]) -> None:
        """Callback to resolve the future safely inside the event loop."""
        if not fut.done():
            fut.set_result(None)
