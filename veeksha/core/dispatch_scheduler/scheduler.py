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

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Priority queue of requests sorted by ready_at time (implemented as min-heap)
        self._ready_queue: List[_ScheduledItem] = []
        # All state for each session, keyed by session_id
        self._sessions: Dict[int, SessionState] = {}
        # Reverse mapping from request_id to (session_id, sequence_index)
        self._id_to_session_seq: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        self._start_monotonic = time.monotonic()

    def _now(self) -> float:
        return time.monotonic() - self._start_monotonic

    def _add_to_ready_queue(self, ready_at: float, request: RequestConfig) -> None:
        """Add a request to the priority queue, sorted by ready time."""
        req_id = request.id if request.id is not None else -1
        heapq.heappush(
            self._ready_queue,
            _ScheduledItem(ready_at=ready_at, request_id=int(req_id), request=request),
        )

    def add_request(self, request: RequestConfig) -> None:
        with self._lock:
            existing = self._id_to_session_seq.get(request.id)
            if existing is not None:
                existing_session_id, existing_seq_idx = existing
                if (
                    existing_session_id == request.session_id
                    and existing_seq_idx == request.session_sequence_index
                ):
                    logger.debug(
                        "Ignoring duplicate add of request_id=%s for session=%s seq=%s",
                        request.id,
                        request.session_id,
                        request.session_sequence_index,
                    )
                    return
            session_id = request.session_id
            seq_idx = request.session_sequence_index

            session = self._sessions.get(session_id)
            if session is None:
                session = SessionState()
                self._sessions[session_id] = session

            # Drop requests for canceled sessions
            if session.is_canceled:
                return

            # Remember cancel policy from any request in the session
            if request.cancel_session_on_failure is not None:
                session.cancel_on_failure = bool(request.cancel_session_on_failure)

            # Track mapping for completion callbacks
            self._id_to_session_seq[request.id] = (session_id, seq_idx)
            session.open_requests += 1

            if seq_idx == 0:
                # must be anchored by absolute timestamp
                if request.session_start_time is not None:
                    ready_at = float(request.session_start_time)
                else:
                    raise ValueError(
                        "session_start_time is required for first-in-session requests"
                    )
                self._add_to_ready_queue(ready_at, request)
            else:
                # Queue until prior is completed; then we can compute ready time
                session.pending_requests[seq_idx] = request
                self._maybe_release_next_locked(session_id)

    def _maybe_release_next_locked(self, session_id: int) -> None:
        # Release next-in-order pending request if its predecessor is completed
        session = self._sessions.get(session_id)
        if not session:
            return

        next_seq = session.completed_sequence + 1
        if next_seq in session.pending_requests:
            req = session.pending_requests.pop(next_seq)
            # compute ready_at using last completion time + delay
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
            session_id, seq_idx = self._id_to_session_seq.pop(request_id, (None, None))
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
                    self._cancel_pending_requests_locked(session_id, session)
            # Mark completion
            if seq_idx > session.completed_sequence:
                session.completed_sequence = seq_idx
            session.last_completion_time = completed_at
            session.open_requests = max(0, session.open_requests - 1)
            # Try releasing next pending
            self._maybe_release_next_locked(session_id)
            self._maybe_garbage_collect_session_locked(session_id)

    def _cancel_pending_requests_locked(
        self, session_id: int, session: SessionState
    ) -> None:
        if not session.pending_requests:
            return
        for pending in session.pending_requests.values():
            self._id_to_session_seq.pop(pending.id, None)
        dropped = len(session.pending_requests)
        session.pending_requests.clear()
        session.open_requests = max(0, session.open_requests - dropped)
        self._maybe_garbage_collect_session_locked(session_id)

    def _maybe_garbage_collect_session_locked(self, session_id: int) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        if session.open_requests > 0:
            return
        if session.pending_requests:
            return
        self._sessions.pop(session_id, None)
