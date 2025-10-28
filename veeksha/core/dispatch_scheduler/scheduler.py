"""Request scheduler for dispatching with session-aware dependencies."""

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from veeksha.core.dispatch_scheduler.session_state import SessionState
from veeksha.core.request_config import RequestConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


@dataclass(order=True)
class _ScheduledItem:
    """Internal representation of a scheduled request."""

    ready_at: float
    request_id: int = field(compare=False)
    request: RequestConfig = field(compare=False)


class DispatchScheduler:
    """Thread-safe scheduler for dispatching requests with session-aware dependencies.

    Time base: seconds since scheduler creation (monotonic reference).
    """

    def __init__(self, max_concurrent_sessions=None) -> None:
        self._lock = threading.Lock()
        # Priority queue of requests sorted by ready_at time (implemented as min-heap)
        self._ready_queue: List[_ScheduledItem] = []
        # All state for each session, keyed by session_id
        self._sessions: Dict[int, SessionState] = {}
        # Reverse mapping from request_id to (session_id, sequence_index)
        self._id_to_session_seq: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        self._start_monotonic = time.monotonic()
        self._non_session_ready_cursor: float = 0.0

        self._max_concurrent_sessions = max_concurrent_sessions
        self._num_active_sessions = 0
        from collections import deque
        self._queued_sessions = deque()
        if self._buffered_mode:
            logger.debug(f"Buffered session mode enabled. size={max_concurrent_sessions}")
        self._num_requests_in_session = {}

    def _now(self) -> float:
        return time.monotonic() - self._start_monotonic
    
    @property
    def _buffered_mode(self) -> bool:
        return self._max_concurrent_sessions is not None

    def _is_session_start(self, request):
        return request.session_sequence_index == 0

    def _mark_session_ready(self, request, immediate=False):
        # First-in-session: anchor by absolute if provided; else treat as normal delay
        if immediate:
            ready_at = self._now()
        elif self._buffered_mode:
            # Abide by QPS if possible
            ready_at = max(self._now(), request.anchor_at_s)
        elif request.anchor_at_s is not None:
            ready_at = float(request.anchor_at_s)
        else:
            ready_at = self._now() + float(request.dispatch_delay)
        logger.debug(f"MARKING SESSION READY {request.session_id}")
        self._add_to_ready_queue(ready_at, request)
        self._num_active_sessions += 1

    def _add_to_ready_queue(self, ready_at: float, request: RequestConfig) -> None:
        """Add a request to the priority queue, sorted by ready time."""
        req_id = request.id if request.id is not None else -1
        heapq.heappush(
            self._ready_queue,
            _ScheduledItem(ready_at=ready_at, request_id=int(req_id), request=request),
        )

    def add_request(self, request: RequestConfig) -> None:
        with self._lock:
            req_id = request.id if (request.id is not None) else -1
            request.id = req_id
            self._id_to_session_seq[req_id] = (
                request.session_id,
                request.session_sequence_index,
            )
            self._num_requests_in_session[request.session_id] = request.num_requests_in_session

            if (request.session_id is not None) and (
                request.session_sequence_index is not None
            ):
                # Get or create session state
                session = self._sessions.setdefault(request.session_id, SessionState())

                # Drop requests for canceled sessions
                if session.is_canceled:
                    return

                # Remember cancel policy from any request in the session
                if request.cancel_session_on_failure is not None:
                    session.cancel_on_failure = bool(request.cancel_session_on_failure)

                if self._is_session_start(request):
                    if self._buffered_mode and self._num_active_sessions >= self._max_concurrent_sessions:
                        self._queued_sessions.append(request)
                    else:
                        self._mark_session_ready(request)
                else:
                    # Queue until prior is completed; then we can compute ready time
                    session.pending_requests[request.session_sequence_index] = request
                    self._maybe_release_next_locked(request.session_id)
            else:
                # Non-session request: schedule by dispatch_delay
                anchor_base = max(self._non_session_ready_cursor, self._now())
                ready_at = anchor_base + float(request.dispatch_delay)
                self._non_session_ready_cursor = ready_at
                self._add_to_ready_queue(ready_at, request)

    def _maybe_release_next_locked(self, session_id) -> None:
        # Release next-in-order pending request if its predecessor is completed
        session = self._sessions.get(session_id)
        if not session:
            return

        next_seq = session.completed_sequence + 1
        if next_seq in session.pending_requests:
            req = session.pending_requests.pop(next_seq)
            # compute ready_at using last completion time + wait_after_prev_response_s
            wait = float(req.wait_after_prev_response_s or 0.0)
            if session.last_completion_time is None:
                # If predecessor completion is unknown, keep it pending
                session.pending_requests[next_seq] = req
                return
            ready_at = session.last_completion_time + wait
            self._add_to_ready_queue(ready_at, req)
            # Try to cascade only one step; caller may call again after completions


    def pop_ready(self) -> Optional[RequestConfig]:
        with self._lock:
            if not self._ready_queue:
                return None
            now = self._now()
            if self._ready_queue[0].ready_at <= now:
                item = heapq.heappop(self._ready_queue)
                return item.request
            return None

    def notify_completion(
        self, request_id: Optional[int], completed_at_monotonic: float, success: bool
    ) -> None:
        if request_id is None:
            return
        with self._lock:
            session_id, seq_idx = self._id_to_session_seq.get(request_id, (None, None))
            if session_id is None or seq_idx is None:
                return

            session = self._sessions.get(session_id)
            if not session:
                return

            # Convert absolute monotonic to scheduler time base
            completed_at = completed_at_monotonic - self._start_monotonic

            if not success:
                if session.cancel_on_failure:
                    # Cancel remaining requests in this session
                    session.is_canceled = True
                    session.pending_requests.clear()
            # Mark completion
            if seq_idx > session.completed_sequence:
                session.completed_sequence = seq_idx
            session.last_completion_time = completed_at
            # Try releasing next pending
            logger.debug(f"REQUEST {request_id} FINISHED of {session_id} ({seq_idx+1}/{self._num_requests_in_session[session_id]})")
            self._maybe_release_next_locked(session_id)
            if self._buffered_mode and (seq_idx+1 >= self._num_requests_in_session[session_id]):
                logger.debug(f"SESSION FINISHED {session_id}")
                self._num_active_sessions -= 1
                if len(self._queued_sessions) > 0:
                    request = self._queued_sessions.popleft()
                    self._mark_session_ready(request, immediate=True)
