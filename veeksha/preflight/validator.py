"""Gate a ScoreReport into a verdict.

Three outcomes, distinguishing *whose* fault a failure is:

* ``PASS``               -- every gate passes; the harness measures faithfully.
* ``FAIL``               -- the harness itself adds drift beyond threshold
                            (delivery lag, dispatch drift, or too many unpaired
                            requests). The benchmark numbers cannot be trusted.
* ``SERVER_AT_CAPACITY`` -- the *mock server* could not hold its own ttfc/tpoc
                            schedule, i.e. the server was the bottleneck, so the
                            harness cannot be certified either way (not the
                            harness's fault; re-run lighter / on a quieter box).

Server-pacing gates are checked first: if the ground-truth generator is jittery,
the delivery numbers built on top of it are meaningless, so that outcome wins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from veeksha.preflight import scorer
from veeksha.preflight.models import ScoreReport

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_SERVER_AT_CAPACITY = "SERVER_AT_CAPACITY"

# Gate categories: "server" gates blame the mock; "harness" gates blame veeksha.
_CATEGORY_SERVER = "server"
_CATEGORY_HARNESS = "harness"


@dataclass
class GateResult:
    name: str
    metric: str
    category: str
    p99: float
    threshold: float
    count: int
    passed: bool


@dataclass
class ValidationResult:
    verdict: str
    gates: List[GateResult] = field(default_factory=list)

    @property
    def is_pass(self) -> bool:
        return self.verdict == VERDICT_PASS

    def failed_gates(self, category: str) -> List[GateResult]:
        return [g for g in self.gates if g.category == category and not g.passed]


def _gate(
    report: ScoreReport,
    name: str,
    metric: str,
    category: str,
    threshold: float,
    optional: bool = False,
) -> GateResult:
    summary = report.metrics.get(metric)
    p99 = summary.p99 if summary is not None else float("nan")
    count = summary.count if summary is not None else 0
    if optional and count == 0:
        # Metric not applicable to this client (e.g. input metrics on a
        # non-streaming-input client) -> N/A, treated as passing.
        return GateResult(name, metric, category, p99, threshold, count, True)
    # A NaN p99 (no data) fails safe: NaN < threshold is False.
    passed = (not math.isnan(p99)) and p99 < threshold
    return GateResult(name, metric, category, p99, threshold, count, passed)


def run_validation(
    report: ScoreReport,
    *,
    delivery_lag_threshold_ms: float,
    server_pacing_threshold_ms: float,
    dispatch_drift_threshold_ms: float,
    input_pacing_threshold_ms: float,
    max_unpaired_fraction: float,
) -> ValidationResult:
    """Apply the gates and derive the verdict."""
    gates: List[GateResult] = [
        # server pacing fidelity (blames the mock)
        _gate(
            report,
            "server ttfc pacing",
            scorer.M_SERVER_TTFC_ABS_ERR,
            _CATEGORY_SERVER,
            server_pacing_threshold_ms,
        ),
        _gate(
            report,
            "server tpoc pacing",
            scorer.M_SERVER_TPOC_ABS_ERR,
            _CATEGORY_SERVER,
            server_pacing_threshold_ms,
        ),
        # request/response delivery lag (blames the harness transport)
        _gate(
            report,
            "request delivery",
            scorer.M_REQUEST_DELIVERY,
            _CATEGORY_HARNESS,
            delivery_lag_threshold_ms,
        ),
        _gate(
            report,
            "response delivery",
            scorer.M_RESPONSE_DELIVERY,
            _CATEGORY_HARNESS,
            delivery_lag_threshold_ms,
        ),
        # end-to-end dispatch drift (blames the scheduler/dispatcher)
        _gate(
            report,
            "dispatch drift",
            scorer.M_LIFECYCLE_READY_TO_SEND,
            _CATEGORY_HARNESS,
            dispatch_drift_threshold_ms,
        ),
        # streaming-input only (optional -> N/A for single-shot clients)
        _gate(
            report,
            "input delivery",
            scorer.M_INPUT_DELIVERY,
            _CATEGORY_HARNESS,
            delivery_lag_threshold_ms,
            optional=True,
        ),
        _gate(
            report,
            "input pacing",
            scorer.M_INPUT_PACING_ABS_ERR,
            _CATEGORY_HARNESS,
            input_pacing_threshold_ms,
            optional=True,
        ),
    ]

    # unpaired-fraction gate (harness dropped or mismatched requests)
    unpaired_ok = report.unpaired_fraction <= max_unpaired_fraction
    gates.append(
        GateResult(
            name="unpaired fraction",
            metric="unpaired_fraction",
            category=_CATEGORY_HARNESS,
            p99=report.unpaired_fraction,
            threshold=max_unpaired_fraction,
            count=report.n_requests,
            passed=unpaired_ok,
        )
    )

    # SERVER_AT_CAPACITY only applies when the server actually produced pacing
    # data that breached -- a missing (count 0) server gate is not the mock
    # being jittery, it means the run itself failed to measure, which is a
    # harness problem (falls through to FAIL below).
    server_failed = any(
        not g.passed and g.count > 0 and g.category == _CATEGORY_SERVER for g in gates
    )
    harness_failed = any(
        not g.passed and g.category == _CATEGORY_HARNESS for g in gates
    )

    if server_failed:
        verdict = VERDICT_SERVER_AT_CAPACITY
    elif harness_failed:
        verdict = VERDICT_FAIL
    else:
        verdict = VERDICT_PASS

    return ValidationResult(verdict=verdict, gates=gates)
