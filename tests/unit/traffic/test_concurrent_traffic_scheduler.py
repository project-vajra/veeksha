"""Unit tests for ConcurrentTrafficScheduler."""

import time
from typing import Dict

import pytest

from veeksha.config.traffic import ConcurrentTrafficConfig
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import SessionEdge, SessionGraph, SessionNode, add_edge, add_node
from veeksha.traffic.concurrent import ConcurrentTrafficScheduler
from veeksha.types import ChannelModality


def wait_until(predicate, timeout_s=0.5, interval_s=0.005):
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def make_request(request_id: int) -> Request:
    return Request(
        id=request_id,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text=f"test_{request_id}"
            )
        },
    )


def make_linear_session(session_id: int, num_requests: int) -> Session:
    graph = SessionGraph()
    requests: Dict[int, Request] = {}
    for i in range(num_requests):
        add_node(graph, SessionNode(id=i, wait_after_ready=0.0))
        requests[i] = make_request(request_id=session_id * 100 + i)
    for i in range(num_requests - 1):
        add_edge(graph, SessionEdge(src=i, dst=i + 1))
    return Session(id=session_id, session_graph=graph, requests=requests)


def make_scheduler(target: int = 2) -> ConcurrentTrafficScheduler:
    config = ConcurrentTrafficConfig(target_concurrent_sessions=target, rampup_seconds=0)
    return ConcurrentTrafficScheduler(config, SeedManager(seed=42))


@pytest.mark.unit
def test_activates_up_to_target() -> None:
    """Sessions up to target are activated immediately."""
    scheduler = make_scheduler(target=2)
    
    scheduler.schedule_session(make_linear_session(1, 1))
    scheduler.schedule_session(make_linear_session(2, 1))
    scheduler.schedule_session(make_linear_session(3, 1))
    
    assert len(scheduler._sessions) == 2
    assert len(scheduler._pending_sessions) == 1


@pytest.mark.unit
def test_pending_activated_on_completion() -> None:
    """Pending session is activated when an active session completes."""
    scheduler = make_scheduler(target=1)
    
    scheduler.schedule_session(make_linear_session(1, 1))
    scheduler.schedule_session(make_linear_session(2, 1))
    
    assert len(scheduler._sessions) == 1
    assert 1 in scheduler._sessions
    
    # pop and complete first session
    req = scheduler.pop_ready()[0]
    assert req is not None
    scheduler.notify_completion(req.id, time.monotonic(), success=True)
    
    # second session should now be active
    assert len(scheduler._sessions) == 1
    assert 2 in scheduler._sessions


@pytest.mark.unit
def test_pending_activated_on_cancel() -> None:
    """Pending session is activated when an active session is cancelled."""
    scheduler = make_scheduler(target=1)
    
    scheduler.schedule_session(make_linear_session(1, 2))
    scheduler.schedule_session(make_linear_session(2, 1))
    
    # pop first request, fail it
    req = scheduler.pop_ready()[0]
    scheduler.notify_completion(req.id, time.monotonic(), success=False)
    
    # session 1 should be cancelled, session 2 activated
    assert 1 not in scheduler._sessions
    assert 2 in scheduler._sessions


@pytest.mark.unit
def test_sessions_start_immediately() -> None:
    """Active sessions start at current time, not scheduled future time."""
    scheduler = make_scheduler(target=2)
    scheduler.schedule_session(make_linear_session(1, 1))
    
    # should be ready immediately
    req = scheduler.pop_ready()[0]
    assert req is not None


@pytest.mark.unit
def test_linear_chain_within_session() -> None:
    """Requests within a session still respect dependencies."""
    scheduler = make_scheduler(target=1)
    scheduler.schedule_session(make_linear_session(1, 2))
    
    # first request ready
    req1 = scheduler.pop_ready()[0]
    assert req1 is not None
    
    # second not ready yet
    assert scheduler.pop_ready() is None
    
    # complete first, second becomes ready
    scheduler.notify_completion(req1.id, time.monotonic(), success=True)
    req2 = scheduler.pop_ready()[0]
    assert req2 is not None


@pytest.mark.unit
def test_history_inheritance() -> None:
    """Requests inherit history from their history parent."""
    scheduler = make_scheduler(target=1)
    
    # Linear: 0 -> 1 -> 2
    session = make_linear_session(session_id=1, num_requests=3)
    scheduler.schedule_session(session)
    
    # 0 ready, no history
    req0 = scheduler.pop_ready()[0]
    assert req0 is not None
    assert req0.history == []
    
    # Complete 0 with history
    # Complete 0 with history
    response_0 = ChannelResponse(
        modality=ChannelModality.TEXT,
        content="response_0"
    )
    scheduler.notify_completion(
        request_id=req0.id, 
        completed_at_monotonic=time.monotonic(), 
        success=True, 
        channel_responses={ChannelModality.TEXT: response_0}
    )
    
    # 1 ready, inherits from 0
    req1 = scheduler.pop_ready()[0]
    assert req1 is not None
    assert len(req1.history) == 2
    assert req1.history[0] == {"role": "user", "content": "test_100"}
    assert req1.history[1] == {"role": "assistant", "content": "response_0"}
    
    # Complete 1 with history
    response_1 = ChannelResponse(
        modality=ChannelModality.TEXT,
        content="response_1"
    )
    scheduler.notify_completion(
        request_id=req1.id, 
        completed_at_monotonic=time.monotonic(), 
        success=True, 
        channel_responses={ChannelModality.TEXT: response_1}
    )
    
    # 2 ready, inherits from 1 (which had history from 0)
    req2 = scheduler.pop_ready()[0]
    assert req2 is not None
    assert len(req2.history) == 4
    assert req2.history[0] == {"role": "user", "content": "test_100"}
    assert req2.history[1] == {"role": "assistant", "content": "response_0"}
    assert req2.history[2] == {"role": "user", "content": "test_101"}
    assert req2.history[3] == {"role": "assistant", "content": "response_1"}


@pytest.mark.unit
def test_ambiguous_history_inheritance() -> None:
    """Raises ValueError if multiple history parents exist."""
    scheduler = make_scheduler(target=1)
    
    # Graph: 0 -> 2, 1 -> 2
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    add_node(graph, SessionNode(id=1, wait_after_ready=0.0))
    add_node(graph, SessionNode(id=2, wait_after_ready=0.0))
    
    # Both are history parents (default is_history_parent=True)
    add_edge(graph, SessionEdge(src=0, dst=2, is_history_parent=True))
    add_edge(graph, SessionEdge(src=1, dst=2, is_history_parent=True))
    
    requests = {
        0: make_request(200),
        1: make_request(201),
        2: make_request(202),
    }
    session = Session(id=2, session_graph=graph, requests=requests)
    scheduler.schedule_session(session)
    
    # Complete parents
    req0 = scheduler.pop_ready()[0]
    scheduler.notify_completion(req0.id, time.monotonic(), success=True)
    
    req1 = scheduler.pop_ready()[0]
    scheduler.notify_completion(req1.id, time.monotonic(), success=True)
    
    # 2 should be ready now, but pop_ready checks history ambiguity
    with pytest.raises(ValueError, match="Ambiguous history inheritance"):
        scheduler.pop_ready()


@pytest.mark.unit
def test_no_history_parent() -> None:
    """Requests with no history parent start with empty history."""
    scheduler = make_scheduler(target=1)
    
    # Graph: 0 -> 1 (but is_history_parent=False)
    graph = SessionGraph()
    add_node(graph, SessionNode(id=0, wait_after_ready=0.0))
    add_node(graph, SessionNode(id=1, wait_after_ready=0.0))
    add_edge(graph, SessionEdge(src=0, dst=1, is_history_parent=False))
    
    requests = {
        0: make_request(300),
        1: make_request(301),
    }
    session = Session(id=3, session_graph=graph, requests=requests)
    scheduler.schedule_session(session)
    
    # Complete 0
    req0 = scheduler.pop_ready()[0]
    assert req0 is not None
    response_0 = ChannelResponse(
        modality=ChannelModality.TEXT,
        content="response_0"
    )
    scheduler.notify_completion(
        request_id=req0.id, 
        completed_at_monotonic=time.monotonic(), 
        success=True, 
        channel_responses={ChannelModality.TEXT: response_0}
    )
    
    # 1 ready, but no history inherited (empty list)
    req1 = scheduler.pop_ready()[0]
    assert req1 is not None
    assert req1.history == []
