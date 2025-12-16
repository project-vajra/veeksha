"""Unit tests for ConcurrentTrafficScheduler."""

import time
from typing import Dict

import pytest

from veeksha.new.config.traffic import ConcurrentTrafficConfig
from veeksha.new.core.request import Request
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.session import Session
from veeksha.new.core.session_graph import SessionEdge, SessionGraph, SessionNode, add_edge, add_node
from veeksha.new.traffic.concurrent import ConcurrentTrafficScheduler
from veeksha.new.types import ChannelModality


def wait_until(predicate, timeout_s=0.5, interval_s=0.005):
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def make_request(request_id: int) -> Request:
    return Request(id=request_id, channels={ChannelModality.TEXT: "test"}, model="dummy")


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
    config = ConcurrentTrafficConfig(target_concurrent_sessions=target)
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
    req = scheduler.pop_ready()
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
    req = scheduler.pop_ready()
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
    req = scheduler.pop_ready()
    assert req is not None


@pytest.mark.unit
def test_linear_chain_within_session() -> None:
    """Requests within a session still respect dependencies."""
    scheduler = make_scheduler(target=1)
    scheduler.schedule_session(make_linear_session(1, 2))
    
    # first request ready
    req1 = scheduler.pop_ready()
    assert req1 is not None
    
    # second not ready yet
    assert scheduler.pop_ready() is None
    
    # complete first, second becomes ready
    scheduler.notify_completion(req1.id, time.monotonic(), success=True)
    req2 = scheduler.pop_ready()
    assert req2 is not None
