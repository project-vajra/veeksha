import time

import pytest

from veeksha.core.dispatch_scheduler import DispatchScheduler
from veeksha.core.request_config import RequestConfig
from veeksha.metrics.metric_store import MetricStore
from veeksha.config.metrics import MetricsConfig
from veeksha.metrics.request_metrics import RequestMetrics


def wait_until(predicate, timeout_s=0.5, interval_s=0.005):
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


@pytest.mark.unit
def test_scheduler_first_in_session_anchor_ready() -> None:
    scheduler = DispatchScheduler()

    req = RequestConfig(
        model="dummy",
        prompt=("", 0),
        id=1,
        session_id=10,
        session_sequence_index=0,
        arrival_time=0.05,
    )

    scheduler.add_request(req)

    # Not ready immediately
    assert scheduler.pop_ready() is None

    # Becomes ready around anchor time
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)


@pytest.mark.unit
def test_scheduler_first_in_session_delay_ready() -> None:
    scheduler = DispatchScheduler()
    
    # Request scheduled relative to now
    delay = 0.05
    req = RequestConfig(
        model="dummy",
        prompt=("", 0),
        id=99,
        session_id=99,
        session_sequence_index=0,
        delay=delay,
    )
    
    scheduler.add_request(req)
    
    # Not ready immediately
    assert scheduler.pop_ready() is None
    
    # Becomes ready after delay
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)


@pytest.mark.unit
def test_scheduler_in_session_waits_for_prev_completion_then_gap() -> None:
    scheduler = DispatchScheduler()

    first = RequestConfig(
        model="dummy",
        prompt=("", 0),
        id=1,
        session_id=42,
        session_sequence_index=0,
        arrival_time=0.01,
    )
    second = RequestConfig(
        model="dummy",
        prompt=("", 0),
        id=2,
        session_id=42,
        session_sequence_index=1,
        delay=0.07,
    )

    scheduler.add_request(first)
    scheduler.add_request(second)

    # First should get ready around its anchor
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)

    # Before completion, second must not be ready
    assert scheduler.pop_ready() is None

    # Notify first completion now
    scheduler.notify_completion(request_id=1, completed_at_monotonic=time.monotonic(), success=True)

    # Second should become ready after wait gap from completion
    start = time.monotonic()
    got = wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)
    assert got, "Second request did not become ready"
    elapsed = time.monotonic() - start
    # Allow small scheduling jitter
    assert elapsed >= 0.06, f"Second released too early: elapsed={elapsed:.3f}s"


@pytest.mark.unit
def test_scheduler_cancel_session_on_failure_drops_pending() -> None:
    scheduler = DispatchScheduler()

    first = RequestConfig(
        model="dummy",
        prompt=("", 0),
        id=11,
        session_id=77,
        session_sequence_index=0,
        arrival_time=0.0,
        cancel_session_on_failure=True,
    )
    second = RequestConfig(
        model="dummy",
        prompt=("", 0),
        id=12,
        session_id=77,
        session_sequence_index=1,
        delay=0.02,
        cancel_session_on_failure=True,
    )

    scheduler.add_request(first)
    scheduler.add_request(second)

    # First should be ready immediately (anchor 0)
    ready_first = wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.2)
    assert ready_first

    # Fail the first request -> should cancel session
    scheduler.notify_completion(request_id=11, completed_at_monotonic=time.monotonic(), success=False)

    # Second must never become ready
    assert scheduler.pop_ready() is None
    time.sleep(0.05)
    assert scheduler.pop_ready() is None


@pytest.mark.unit
def test_request_level_metrics_record_errored_and_successful_requests() -> None:
    # Ensure MetricStore persists request-level arrays for only successful requests
    ms = MetricStore(timeout=10, max_requests=10, metrics_config=MetricsConfig())

    # One successful request
    ok = RequestMetrics(
        request_dispatched_at=0.01,
        inter_token_times=[0.02, 0.03],
        num_prompt_tokens=10,
        num_output_tokens=2,
        error_msg=None,
        error_code=None,
        request_id=0,
    )
    ms.add_request_metrics(ok)

    # One errored request; should not appear in request-level arrays
    err = RequestMetrics(
        request_dispatched_at=0.05,
        inter_token_times=[],
        num_prompt_tokens=8,
        num_output_tokens=0,
        error_msg="boom",
        error_code=500,
        request_id=1,
    )
    ms.add_request_metrics(err)

    rlm = ms.request_level_metrics
    assert len(rlm.request_dispatched_at) == 1
    assert rlm.request_dispatched_at[0] == 0.01
    
    
@pytest.mark.unit
def test_scheduler_interleaved_sessions() -> None:
    """Test that two independent sessions can be scheduled concurrently without interference."""
    scheduler = DispatchScheduler()

    # Session A: Starts at t=0.01, 2nd req after 0.05s delay
    sA_req1 = RequestConfig(model="m", prompt=("", 0), id=10, session_id=100, session_sequence_index=0, arrival_time=0.01)
    sA_req2 = RequestConfig(model="m", prompt=("", 0), id=11, session_id=100, session_sequence_index=1, delay=0.05)

    # Session B: Starts at t=0.02, 2nd req after 0.05s delay
    sB_req1 = RequestConfig(model="m", prompt=("", 0), id=20, session_id=200, session_sequence_index=0, arrival_time=0.02)
    sB_req2 = RequestConfig(model="m", prompt=("", 0), id=21, session_id=200, session_sequence_index=1, delay=0.05)

    scheduler.add_request(sA_req1)
    scheduler.add_request(sA_req2)
    scheduler.add_request(sB_req1)
    scheduler.add_request(sB_req2)

    # Both first requests become ready near start
    r1 = wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)
    r2 = wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)
    assert r1 and r2

    # Now both sessions are waiting for completion.
    # Complete Session A's first request.
    scheduler.notify_completion(request_id=10, completed_at_monotonic=time.monotonic(), success=True)
    
    # Session A's second request should become ready after delay
    rA2 = wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)
    assert rA2, "Session A second request failed to become ready"
    
    # Session B's second request should still be pending
    assert scheduler.pop_ready() is None

    # Complete Session B's first request
    scheduler.notify_completion(request_id=20, completed_at_monotonic=time.monotonic(), success=True)

    # Session B's second request becomes ready
    rB2 = wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)
    assert rB2, "Session B second request failed to become ready"


@pytest.mark.unit
def test_scheduler_out_of_order_arrival() -> None:
    """Test adding requests for a session in non-sequential order."""
    scheduler = DispatchScheduler()
    
    # Req 2 (dependent) arrives before Req 1 (start)
    req2 = RequestConfig(model="m", prompt=("", 0), id=2, session_id=50, session_sequence_index=1, delay=0.01)
    req1 = RequestConfig(model="m", prompt=("", 0), id=1, session_id=50, session_sequence_index=0, arrival_time=0.01)
    
    scheduler.add_request(req2)
    scheduler.add_request(req1)
    
    # Req 1 should become ready
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)
    
    # Complete Req 1
    scheduler.notify_completion(request_id=1, completed_at_monotonic=time.monotonic(), success=True)
    
    # Req 2 should become ready
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)


@pytest.mark.unit
def test_scheduler_cancellation_cascade() -> None:
    """Test that failure cancels ALL subsequent requests in the session."""
    scheduler = DispatchScheduler()
    
    # Chain of 3 requests
    r1 = RequestConfig(model="m", prompt=("", 0), id=1, session_id=9, session_sequence_index=0, arrival_time=0.0, cancel_session_on_failure=True)
    r2 = RequestConfig(model="m", prompt=("", 0), id=2, session_id=9, session_sequence_index=1, delay=0.0)
    r3 = RequestConfig(model="m", prompt=("", 0), id=3, session_id=9, session_sequence_index=2, delay=0.0)
    
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    scheduler.add_request(r3)
    
    # R1 ready
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.2)
    
    # Fail R1
    scheduler.notify_completion(request_id=1, completed_at_monotonic=time.monotonic(), success=False)
    
    # Ensure R2 AND R3 are never scheduled
    time.sleep(0.05)
    assert scheduler.pop_ready() is None
    
    # Even if we hypothetically completed R2 (impossible flow), R3 shouldn't appear
    scheduler.notify_completion(request_id=2, completed_at_monotonic=time.monotonic(), success=True)
    assert scheduler.pop_ready() is None


