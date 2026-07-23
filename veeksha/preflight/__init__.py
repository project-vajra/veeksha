"""Preflight timing-fidelity validation.

Before trusting a benchmark's numbers we certify that the *harness itself* keeps
time honestly: that requests are dispatched when the schedule says, that they
reach the server when the client thinks they were sent, and that streamed
responses arrive when the server says it sent them. We do this by pointing the
real scheduler -> dispatch -> client path at deterministic mock servers (run in
separate processes to keep their emit schedules out of the clients' way) that
stamp ground-truth send/receive times, then scoring the drift between what each
side observed.

Notation used throughout (all ``time.monotonic()``):

* ``t_cs``   -- client-sent: request handed to the transport
* ``t_sr``   -- server-received: mock server accepted the request
* ``t_ss_i`` -- server-sent: mock emitted response chunk ``i``
* ``t_cr_i`` -- client-received: client observed response chunk ``i``

Drifts we track as metrics (p50/p99, gated on thresholds):

* request-delivery lag   ``t_sr - t_cs``
* response-delivery lag  ``t_cr_i - t_ss_i``
* server pacing fidelity ``t_ss_1 - t_sr`` vs ttfc, ``t_ss_{i+1} - t_ss_i`` vs tpoc
* dispatch drift         ``scheduler_dispatched_at - scheduler_ready_at`` and
  the end-to-end ``t_cs - scheduler_ready_at``
"""

from veeksha.preflight.models import (
    MetricSummary,
    ScoreReport,
    ServerRequestRecord,
)
from veeksha.preflight.scorer import percentile, score, summarize

__all__ = [
    "MetricSummary",
    "ScoreReport",
    "ServerRequestRecord",
    "percentile",
    "score",
    "summarize",
]
