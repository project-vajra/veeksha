"""Plain data structures shared across the preflight package.

Kept dependency-free (no I/O, no veeksha.core imports beyond typing) so the
scorer and its tests can construct them directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ServerRequestRecord:
    """Ground-truth timestamps a mock server recorded for one request.

    All timestamps are ``time.monotonic()`` from the *server* process. Because
    ``time.monotonic()`` maps to a machine-wide clock (CLOCK_MONOTONIC / mach
    absolute time), these are directly comparable to the client's stamps as long
    as both run on the same host -- which the preflight harness guarantees.
    """

    request_id: int
    server_recv_time: float  # t_sr (connection / request accept)
    server_send_times: List[float] = field(default_factory=list)  # t_ss_i
    # t_sr_i: per-input-segment receipt times. Only populated by streaming-input
    # mocks (realtime_tts / vajra / stt); empty for single-shot HTTP requests.
    input_recv_times: List[float] = field(default_factory=list)
    # When the server began emitting the response. The ttfc pacing metric is
    # anchored here, so it isn't polluted by a streaming-input phase. HTTP mocks
    # leave this None (the scorer falls back to server_recv_time, since they
    # respond immediately); WS mocks set it after the input completes.
    response_start_time: Optional[float] = None

    def to_json(self) -> dict:
        return {
            "request_id": self.request_id,
            "server_recv_time": self.server_recv_time,
            "server_send_times": self.server_send_times,
            "input_recv_times": self.input_recv_times,
            "response_start_time": self.response_start_time,
        }

    @classmethod
    def from_json(cls, d: dict) -> "ServerRequestRecord":
        response_start_time = d.get("response_start_time")
        return cls(
            request_id=int(d["request_id"]),
            server_recv_time=float(d["server_recv_time"]),
            server_send_times=[float(x) for x in d.get("server_send_times", [])],
            input_recv_times=[float(x) for x in d.get("input_recv_times", [])],
            response_start_time=(
                None if response_start_time is None else float(response_start_time)
            ),
        )


@dataclass
class MetricSummary:
    """Distribution summary for one drift metric (all values in ms)."""

    name: str
    count: int
    p50: float
    p99: float
    mean: float
    minimum: float
    maximum: float

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "p50": self.p50,
            "p99": self.p99,
            "mean": self.mean,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class ScoreReport:
    """All drift metrics for one preflight run, plus bookkeeping."""

    metrics: Dict[str, MetricSummary] = field(default_factory=dict)
    n_requests: int = 0
    n_paired_requests: int = 0  # had both client + server records
    unpaired_fraction: float = 0.0

    def to_json(self) -> dict:
        return {
            "n_requests": self.n_requests,
            "n_paired_requests": self.n_paired_requests,
            "unpaired_fraction": self.unpaired_fraction,
            "metrics": {k: v.to_json() for k, v in self.metrics.items()},
        }
