"""Preflight timing-fidelity validation.

Before trusting a benchmark's numbers we certify that the *harness itself* keeps
time honestly: that requests are dispatched when the schedule says, that they
reach the server when the client thinks they were sent, and that streamed
responses arrive when the server says it sent them. We do this by pointing the
real scheduler -> dispatch -> client path at deterministic mock servers (run in
separate processes to keep their emit schedules out of the clients' way) that
stamp ground-truth send/receive times, then scoring the drift between what each
side observed.

The two record books (all times ``time.monotonic()``, joined by request id):

* client, on ``RequestResult`` -- ``client_sent_at`` (request handed to the
  transport) and ``chunk_recv_times[i]`` (response chunk ``i`` observed).
* server, on ``ServerRequestRecord`` -- ``server_recv_time`` (request accepted)
  and ``server_send_times[i]`` (response chunk ``i`` emitted).

Streaming-input clients add ``input_send_times[i]`` / ``input_recv_times[i]``
for the per-segment input phase.

Drifts we track as metrics (p50/p99, gated on thresholds):

* request-delivery lag   ``server_recv_time - client_sent_at``
* response-delivery lag  ``chunk_recv_times[i] - server_send_times[i]``
* server pacing fidelity first ``server_send_times`` gap from the response start
  vs ttfc, and successive gaps vs tpoc
* dispatch drift         ``scheduler_dispatched_at - scheduler_ready_at`` and
  the end-to-end ``client_sent_at - scheduler_ready_at``
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
