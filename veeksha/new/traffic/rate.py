"""Rate-based traffic scheduler for dispatching sessions at a specified rate."""

import heapq
import threading
import time
from typing import Dict, List, Optional, Tuple

from veeksha.new.config.traffic import RateTrafficConfig
from veeksha.new.core.request import Request
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.session import Session
from veeksha.new.core.session_graph import children, parents, ready_at
from veeksha.new.generator.interval.registry import IntervalGeneratorRegistry
from veeksha.new.traffic.base import BaseTrafficScheduler
from veeksha.new.traffic.session_state import ScheduledItem, ScheduledSessionState


class RateTrafficScheduler(BaseTrafficScheduler):
    """Scheduler for dispatching sessions at a specified rate.

    Time base: seconds since scheduler creation (monotonic reference).
    """

    def __init__(self, config: RateTrafficConfig, seed_manager: SeedManager):
        super().__init__(config, seed_manager)
        self._interval_gen = IntervalGeneratorRegistry.get(
            config.interval_generator.get_type(),
            config.interval_generator,
            rng=seed_manager.numpy_factory("interval"),
        )
        self._lock = threading.Lock()
        self._next_start_time: float = 0.0
        self._start_monotonic = time.monotonic()
        self._ready_queue: List[ScheduledItem] = []
        self._sessions: Dict[int, ScheduledSessionState] = {}
        self._request_to_session: Dict[int, Tuple[int, int]] = {}

    def _now(self) -> float:
        return time.monotonic() - self._start_monotonic

    def _add_to_ready_queue(self, ready_at_time: float, request: Request) -> None:
        heapq.heappush(
            self._ready_queue,
            ScheduledItem(
                ready_at=ready_at_time, request_id=request.id, request=request
            ),
        )

    def schedule_session(self, session: Session) -> None:
        with self._lock:
            start_time = self._next_start_time
            self._next_start_time += self._interval_gen.get_next_interval()

            state = ScheduledSessionState(
                session=session,
                session_start_time=start_time,
                completions={},
                pending_nodes=set(session.session_graph.nodes.keys()),
                queued_nodes=set(),
                cancel_on_failure=session.cancel_session_on_failure,
            )
            self._sessions[session.id] = state

            # queue root nodes
            graph = session.session_graph
            for node_id in list(state.pending_nodes):
                if not parents(graph, node_id):
                    node_ready_at = start_time + graph.nodes[node_id].wait_after_ready
                    request = session.requests[node_id]
                    self._add_to_ready_queue(node_ready_at, request)
                    self._request_to_session[request.id] = (session.id, node_id)
                    state.pending_nodes.discard(node_id)
                    state.queued_nodes.add(node_id)

    def pop_ready(self) -> Optional[Request]:
        with self._lock:
            if not self._ready_queue:
                return None
            if self._ready_queue[0].ready_at <= self._now():
                return heapq.heappop(self._ready_queue).request
            return None

    def notify_completion(
        self, request_id: int, completed_at_monotonic: float, success: bool
    ) -> None:
        with self._lock:
            session_id, node_id = self._request_to_session.pop(request_id)
            state = self._sessions[session_id]
            completed_at = completed_at_monotonic - self._start_monotonic

            state.completions[node_id] = completed_at
            state.queued_nodes.discard(node_id)

            # cancel session on failure
            if not success and state.cancel_on_failure:
                state.is_canceled = True
                state.pending_nodes.clear()
                if not state.queued_nodes:
                    del self._sessions[session_id]
                return

            # children might be ready; release them
            graph = state.session.session_graph
            for edge in children(graph, node_id):
                child_id = edge.dst
                if child_id not in state.pending_nodes:
                    continue
                node_ready_at = ready_at(graph, child_id, state.completions)
                if node_ready_at is not None:
                    request = state.session.requests[child_id]
                    self._add_to_ready_queue(node_ready_at, request)
                    self._request_to_session[request.id] = (session_id, child_id)
                    state.pending_nodes.discard(child_id)
                    state.queued_nodes.add(child_id)

            # session is complete
            if not state.pending_nodes and not state.queued_nodes:
                del self._sessions[session_id]
