"""Pure drift math over paired client/server timestamps.

No I/O and no framework imports: everything here operates on plain numbers and
the light dataclasses in :mod:`veeksha.preflight.models`, so it is trivially
unit-testable. ``score`` consumes ``RequestResult``-shaped objects (duck-typed
via attribute access) plus the server's ground-truth records.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, Iterable, List, Mapping, Optional

from veeksha.preflight.models import MetricSummary, ScoreReport, ServerRequestRecord

# Metric names -- referenced by the validator's gates so they stay in sync.
M_REQUEST_DELIVERY = "request_delivery_ms"  # t_sr - t_cs
M_RESPONSE_DELIVERY = "response_delivery_ms"  # t_cr_i - t_ss_i
M_SERVER_TTFC_ABS_ERR = "server_ttfc_abs_error_ms"  # |(t_ss_0 - t_sr) - ttfc|
M_SERVER_TPOC_ABS_ERR = "server_tpoc_abs_error_ms"  # |gap - tpoc|
M_CLIENT_TTFC = "client_observed_ttfc_ms"  # t_cr_0 - t_cs (informational)
M_CLIENT_TPOC = "client_observed_tpoc_ms"  # t_cr_{i+1} - t_cr_i (informational)
M_LIFECYCLE_READY_TO_DISPATCH = "lifecycle_ready_to_dispatch_ms"
M_LIFECYCLE_DISPATCH_TO_PICKUP = "lifecycle_dispatch_to_pickup_ms"
M_LIFECYCLE_PICKUP_TO_SEND = "lifecycle_pickup_to_send_ms"
M_LIFECYCLE_READY_TO_SEND = "lifecycle_ready_to_send_ms"  # end-to-end (t_cs - ready)
# Streaming-input clients only (realtime_tts / vajra / stt):
M_INPUT_DELIVERY = "input_delivery_ms"  # t_sr_i - t_cs_i (per input segment)
M_INPUT_PACING_ABS_ERR = "input_pacing_abs_error_ms"  # |t_cs_i - deadline_i|

_MS = 1000.0


def percentile(values: List[float], q: float) -> float:
    """Linear-interpolation percentile (like numpy's default). NaN if empty."""
    xs = sorted(v for v in values if not math.isnan(v))
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    rank = (q / 100.0) * (len(xs) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[int(lo)]
    frac = rank - lo
    return xs[int(lo)] * (1.0 - frac) + xs[int(hi)] * frac


def summarize(name: str, values: Iterable[float]) -> MetricSummary:
    xs = [v for v in values if not math.isnan(v)]
    if not xs:
        nan = float("nan")
        return MetricSummary(name, 0, nan, nan, nan, nan, nan, nan)
    return MetricSummary(
        name=name,
        count=len(xs),
        p50=percentile(xs, 50),
        p99=percentile(xs, 99),
        mean=sum(xs) / len(xs),
        stdev=statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        minimum=min(xs),
        maximum=max(xs),
    )


def _append(bucket: Dict[str, List[float]], name: str, value: float) -> None:
    bucket.setdefault(name, []).append(value)


def score(
    results: Iterable[object],
    server_records: Mapping[int, ServerRequestRecord],
    ttfc_ms: float,
    tpoc_ms: float,
) -> ScoreReport:
    """Compute every drift metric for one preflight run.

    Args:
        results: RequestResult-shaped objects (need ``request_id`` plus the
            lifecycle/preflight timestamp attributes; missing ones are skipped).
        server_records: request_id -> ServerRequestRecord ground truth.
        ttfc_ms: configured server first-chunk delay, for pacing-fidelity error.
        tpoc_ms: configured server inter-chunk delay, for pacing-fidelity error.
    """
    buckets: Dict[str, List[float]] = {}
    n_requests = 0
    n_paired = 0

    for r in results:
        n_requests += 1
        request_id = getattr(r, "request_id", None)
        client_send_time = getattr(r, "client_sent_at", None)  # t_cs
        ready_time = getattr(r, "scheduler_ready_at", None)
        dispatched_time = getattr(r, "scheduler_dispatched_at", None)
        pickup_time = getattr(r, "client_picked_up_at", None)
        client_recv_times: Optional[List[float]] = getattr(
            r, "chunk_recv_times", None
        )  # t_cr_i
        input_send_times: Optional[List[float]] = getattr(
            r, "input_send_times", None
        )  # t_cs_i
        input_send_deadlines: Optional[List[float]] = getattr(
            r, "input_send_deadlines", None
        )

        server_record = (
            server_records.get(request_id) if request_id is not None else None
        )

        # --- dispatch drift (harness-only; no server record needed) ---
        if ready_time is not None and dispatched_time is not None:
            _append(
                buckets,
                M_LIFECYCLE_READY_TO_DISPATCH,
                (dispatched_time - ready_time) * _MS,
            )
        if dispatched_time is not None and pickup_time is not None:
            _append(
                buckets,
                M_LIFECYCLE_DISPATCH_TO_PICKUP,
                (pickup_time - dispatched_time) * _MS,
            )
        if pickup_time is not None and client_send_time is not None:
            _append(
                buckets,
                M_LIFECYCLE_PICKUP_TO_SEND,
                (client_send_time - pickup_time) * _MS,
            )
        if ready_time is not None and client_send_time is not None:
            _append(
                buckets,
                M_LIFECYCLE_READY_TO_SEND,
                (client_send_time - ready_time) * _MS,
            )

        # --- response delivery: client receipt (t_cr_i) vs server send (t_ss_i) ---
        # Joined from the two record books by index within this request (relies
        # on 1:1 chunk ordering, which the paced localhost mocks hold).
        if (
            client_recv_times
            and server_record is not None
            and server_record.server_send_times
        ):
            for client_recv_time, server_send_time in zip(
                client_recv_times, server_record.server_send_times
            ):
                _append(
                    buckets,
                    M_RESPONSE_DELIVERY,
                    (client_recv_time - server_send_time) * _MS,
                )

        # --- streaming-input pacing accuracy (client-only): t_cs_i vs deadline ---
        if input_send_times and input_send_deadlines:
            for send_time, deadline in zip(input_send_times, input_send_deadlines):
                _append(
                    buckets, M_INPUT_PACING_ABS_ERR, abs(send_time - deadline) * _MS
                )

        # --- client-observed ttfc / tpoc (informational) ---
        if client_recv_times and client_send_time is not None:
            _append(
                buckets, M_CLIENT_TTFC, (client_recv_times[0] - client_send_time) * _MS
            )
        if client_recv_times and len(client_recv_times) >= 2:
            for i in range(1, len(client_recv_times)):
                _append(
                    buckets,
                    M_CLIENT_TPOC,
                    (client_recv_times[i] - client_recv_times[i - 1]) * _MS,
                )

        # --- server-side ground truth: request delivery + pacing fidelity ---
        if server_record is None:
            continue
        n_paired += 1

        server_recv_time = server_record.server_recv_time  # t_sr
        if client_send_time is not None:
            _append(
                buckets,
                M_REQUEST_DELIVERY,
                (server_recv_time - client_send_time) * _MS,
            )

        # streaming-input delivery: server receipt (t_sr_i) vs client send (t_cs_i)
        if input_send_times and server_record.input_recv_times:
            for send_time, server_input_recv in zip(
                input_send_times, server_record.input_recv_times
            ):
                _append(
                    buckets, M_INPUT_DELIVERY, (server_input_recv - send_time) * _MS
                )

        server_send_times = server_record.server_send_times  # t_ss_i
        if server_send_times:
            # Anchor ttfc at when the server began responding (== t_sr for
            # single-shot HTTP; after the input phase for streaming-input WS).
            response_start = server_record.response_start_time
            if response_start is None:
                response_start = server_recv_time
            ttfc_actual = (server_send_times[0] - response_start) * _MS
            _append(buckets, M_SERVER_TTFC_ABS_ERR, abs(ttfc_actual - ttfc_ms))
            for i in range(1, len(server_send_times)):
                gap = (server_send_times[i] - server_send_times[i - 1]) * _MS
                _append(buckets, M_SERVER_TPOC_ABS_ERR, abs(gap - tpoc_ms))

    report = ScoreReport(
        metrics={name: summarize(name, vals) for name, vals in buckets.items()},
        n_requests=n_requests,
        n_paired_requests=n_paired,
        unpaired_fraction=(
            0.0 if n_requests == 0 else (n_requests - n_paired) / n_requests
        ),
    )
    return report
