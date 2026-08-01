"""Unit tests for the pure drift scorer."""

import math
from types import SimpleNamespace

import pytest

from veeksha.preflight import scorer
from veeksha.preflight.models import ServerRequestRecord


def _result(
    request_id,
    ready_time,
    dispatched_time,
    pickup_time,
    client_send_time,
    client_recv_times,
):
    return SimpleNamespace(
        request_id=request_id,
        scheduler_ready_at=ready_time,
        scheduler_dispatched_at=dispatched_time,
        client_picked_up_at=pickup_time,
        client_sent_at=client_send_time,
        chunk_recv_times=client_recv_times,
    )


def test_percentile_interpolates():
    assert scorer.percentile([10.0], 99) == 10.0
    assert scorer.percentile([0.0, 10.0], 50) == pytest.approx(5.0)
    assert scorer.percentile([0.0, 100.0], 99) == pytest.approx(99.0)
    assert math.isnan(scorer.percentile([], 50))


def test_summarize_empty_is_zero_count():
    s = scorer.summarize("x", [])
    assert s.count == 0
    assert math.isnan(s.p99)


def test_score_computes_all_drifts():
    ttfc_ms, tpoc_ms = 200.0, 20.0
    ready_time = 100.0
    dispatched_time = ready_time + 0.001  # +1ms
    pickup_time = dispatched_time + 0.0005  # +0.5ms
    client_send_time = pickup_time + 0.0005  # +0.5ms -> ready_to_send = 2ms
    server_recv_time = client_send_time + 0.0005  # request delivery 0.5ms
    # server emits: ttfc after receipt, then two tpoc gaps -- perfectly on schedule
    server_send_times = [
        server_recv_time + 0.200,
        server_recv_time + 0.220,
        server_recv_time + 0.240,
    ]
    client_recv_times = [t + 0.0003 for t in server_send_times]  # +0.3ms each

    results = [
        _result(
            1,
            ready_time,
            dispatched_time,
            pickup_time,
            client_send_time,
            client_recv_times,
        )
    ]
    server_records = {
        1: ServerRequestRecord(
            1,
            server_recv_time,
            list(server_send_times),
        )
    }

    rep = scorer.score(results, server_records, ttfc_ms, tpoc_ms)
    m = rep.metrics

    assert rep.n_requests == 1
    assert rep.n_paired_requests == 1
    assert rep.unpaired_fraction == 0.0

    assert m[scorer.M_LIFECYCLE_READY_TO_DISPATCH].p50 == pytest.approx(1.0, abs=1e-6)
    assert m[scorer.M_LIFECYCLE_DISPATCH_TO_PICKUP].p50 == pytest.approx(0.5, abs=1e-6)
    assert m[scorer.M_LIFECYCLE_PICKUP_TO_SEND].p50 == pytest.approx(0.5, abs=1e-6)
    assert m[scorer.M_LIFECYCLE_READY_TO_SEND].p50 == pytest.approx(2.0, abs=1e-6)
    assert m[scorer.M_REQUEST_DELIVERY].p50 == pytest.approx(0.5, abs=1e-6)
    assert m[scorer.M_RESPONSE_DELIVERY].count == 3
    assert m[scorer.M_RESPONSE_DELIVERY].p99 == pytest.approx(0.3, abs=1e-6)
    assert m[scorer.M_SERVER_TTFC_ABS_ERR].p99 == pytest.approx(0.0, abs=1e-6)
    assert m[scorer.M_SERVER_TPOC_ABS_ERR].count == 2
    assert m[scorer.M_SERVER_TPOC_ABS_ERR].p99 == pytest.approx(0.0, abs=1e-6)
    assert m[scorer.M_CLIENT_TTFC].p50 == pytest.approx(200.8, abs=1e-3)
    # client-observed tpoc: two gaps of ~20ms
    assert m[scorer.M_CLIENT_TPOC].count == 2
    assert m[scorer.M_CLIENT_TPOC].p50 == pytest.approx(20.0, abs=1e-3)


def test_score_flags_server_pacing_error():
    # Server runs 5ms late on ttfc and 3ms long on every tpoc gap.
    ttfc_ms, tpoc_ms = 200.0, 20.0
    server_recv_time = 50.0
    server_send_times = [
        server_recv_time + 0.205,  # +5ms ttfc
        server_recv_time + 0.228,  # +23ms gap
        server_recv_time + 0.251,  # +23ms gap
    ]
    r = _result(7, None, None, None, None, None)
    rep = scorer.score(
        [r],
        {7: ServerRequestRecord(7, server_recv_time, list(server_send_times))},
        ttfc_ms,
        tpoc_ms,
    )
    assert rep.metrics[scorer.M_SERVER_TTFC_ABS_ERR].p50 == pytest.approx(5.0, abs=1e-6)
    assert rep.metrics[scorer.M_SERVER_TPOC_ABS_ERR].p99 == pytest.approx(3.0, abs=1e-6)


def test_score_unpaired_requests_counted():
    r1 = _result(1, 10.0, 10.001, 10.0015, 10.002, None)
    r2 = _result(2, 10.0, 10.001, 10.0015, 10.002, None)
    # only request 1 has a server record
    rep = scorer.score([r1, r2], {1: ServerRequestRecord(1, 10.0025, [])}, 200.0, 20.0)
    assert rep.n_requests == 2
    assert rep.n_paired_requests == 1
    assert rep.unpaired_fraction == pytest.approx(0.5)
