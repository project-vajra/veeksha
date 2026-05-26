import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, cast

import numpy as np
import pandas as pd

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.logger import init_logger
from veeksha.types import IntervalGeneratorType, TrafficType

logger = init_logger(__name__)


@dataclass
class TestResult:
    summary: Dict[str, Any]
    passed: bool


class HealthChecker:
    def __init__(
        self,
        trace_file: str,
        metrics_file: str,
        benchmark_config: BenchmarkConfig,
    ):
        self.trace_file = trace_file
        self.metrics_file = metrics_file
        self.config = benchmark_config
        self.trace_df = pd.DataFrame()
        self.metrics_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()

    def _get_col(self, col: str) -> str:
        """Get the actual column name after merge (handles suffixed columns)."""
        if col in self.merged_df.columns:
            return col
        # prefer trace over metrics
        if f"{col}_trace" in self.merged_df.columns:
            return f"{col}_trace"
        if f"{col}_metrics" in self.merged_df.columns:
            return f"{col}_metrics"
        return col

    def load_data(self) -> bool:
        try:
            self.metrics_df = pd.read_json(self.metrics_file, lines=True)
            if self.metrics_df.empty:
                logger.warning("Metrics file is empty.")
                return False

            try:
                if os.path.exists(self.trace_file):
                    self.trace_df = pd.read_json(self.trace_file, lines=True)
                else:
                    logger.warning(f"Trace file not found: {self.trace_file}")
                    self.trace_df = pd.DataFrame()
            except Exception as e:
                logger.warning(f"Failed to load trace file; proceeding without it: {e}")
                self.trace_df = pd.DataFrame()

            if not self.trace_df.empty:
                self.merged_df = pd.merge(
                    self.trace_df,
                    self.metrics_df,
                    on="request_id",
                    how="inner",
                    suffixes=("_trace", "_metrics"),
                )
            else:
                self.merged_df = self.metrics_df.copy()

            return True
        except Exception as e:
            logger.error(f"Failed to load health check data: {e}")
            return False

    def run_checks(self) -> List[TestResult]:
        if self.merged_df.empty:
            if not self.load_data():
                return []

        results = []

        # common checks
        results.append(self.check_intra_session_request_arrival())

        # scheduler-specific checks
        traffic_type = self.config.traffic_scheduler.get_type()
        if traffic_type == TrafficType.CONCURRENT:
            results.append(self.check_session_concurrency())
        elif traffic_type == TrafficType.RATE:
            results.append(self.check_session_dispatch_rate())

        # prompt length check
        results.append(self.check_prompt_length())

        # output length check
        results.append(self.check_output_length())

        # lifecycle timing delays check
        results.append(self.check_lifecycle_timing_delays())

        return results

    def run_and_save(self, output_path: str) -> List[TestResult]:
        """Run all health checks and save formatted results to a file.

        Args:
            output_path: Path to save the health check results.

        Returns:
            List of TestResult objects.
        """
        results = self.run_checks()

        with open(output_path, "w") as f:
            for result in results:
                f.write("=" * 60 + "\n")
                f.write(f"{result.summary['name'].upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Result: {'PASSED' if result.passed else 'FAILED'}\n\n")
                for section in result.summary["sections"]:
                    f.write(f"{section['title']}:\n")
                    for k, v in section["results"].items():
                        f.write(f"  {k:30s} {v}\n")
                    f.write("\n")
                f.write("\n")

        logger.info(f"Health checks completed. Results saved to {output_path}")
        return results

    def check_intra_session_request_arrival(self) -> TestResult:
        """
        Verify that requests are dispatched only after their dependencies are met
        plus the specified wait_after_ready time.

        Requires trace data for dependency graphs (session_context).
        """
        if self.trace_df.empty:
            trace_exists = os.path.exists(self.trace_file)
            reason = (
                f"Trace file '{self.trace_file}' not found."
                if not trace_exists
                else f"Trace file '{self.trace_file}' is empty."
            )
            resolution = (
                "Enable the trace recorder or provide a dispatch trace to capture "
                "session dependency data required for this check."
            )
            return TestResult(
                summary={
                    "name": "Intra-Session Request Arrival Check",
                    "sections": [
                        {
                            "title": "Trace Required",
                            "results": {
                                "Status": "Skipped",
                                "Reason": reason,
                                "Resolution": resolution,
                            },
                        }
                    ],
                },
                passed=True,
            )

        deviations: List[float] = []
        violations: List[Dict[str, Any]] = []

        session_col = self._get_col("session_id")
        dispatched_col = self._get_col("scheduler_dispatched_at")

        late_threshold = 5.0  # dispatched more than 5 seconds after ready

        for session_id, group in self.merged_df.groupby(session_col):
            req_completion_map = {
                row["request_id"]: row[self._get_col("client_completed_at")]
                for _, row in group.iterrows()
            }

            node_to_request: Dict[Any, int] = {}
            for _, row in group.iterrows():
                ctx = row.get(self._get_col("session_context"), {})
                if ctx and "node_id" in ctx:
                    node_to_request[ctx["node_id"]] = row["request_id"]  # type: ignore

            for _, row in group.iterrows():
                ctx = row.get(self._get_col("session_context"), {})
                if not ctx:
                    continue

                wait_after_ready = ctx["wait_after_ready"]
                parent_nodes = ctx["parent_nodes"]
                session_size = row.get(self._get_col("session_total_requests"))

                parent_finish_times = []
                for node_id in parent_nodes:
                    req_id = node_to_request.get(node_id)
                    if req_id is not None and req_id in req_completion_map:
                        parent_finish_times.append(req_completion_map[req_id])

                if not parent_finish_times:
                    continue

                ready_at = max(parent_finish_times) + wait_after_ready
                actual_dispatch = row[dispatched_col]
                diff = actual_dispatch - ready_at
                deviations.append(diff)

                if diff > late_threshold:
                    violations.append(
                        {
                            "session_id": session_id,
                            "session_size": session_size,
                            "request_id": row["request_id"],
                            "diff": diff,
                            "ready_at": ready_at,
                            "scheduler_dispatched_at": actual_dispatch,
                        }
                    )

        deviations_array = np.array(deviations) if deviations else np.array([])
        passed = len(violations) == 0

        if deviations_array.size > 0:
            stats = {
                "Requests w/ Dependencies": str(len(deviations)),
                "Min": f"{np.min(deviations_array):.4f}s",
                "Mean": f"{np.mean(deviations_array):.4f}s",
                "Median": f"{np.median(deviations_array):.4f}s",
                "P95": f"{np.percentile(deviations_array, 95):.4f}s",
                "P99": f"{np.percentile(deviations_array, 99):.4f}s",
                "Max": f"{np.max(deviations_array):.4f}s",
                "Std Dev": f"{np.std(deviations_array):.4f}s",
            }
        else:
            stats = {"Requests w/ Dependencies": "0"}

        summary = {
            "name": "Intra-Session Request Arrival Check",
            "sections": [
                {
                    "title": "Description",
                    "results": {
                        "Metric": "Scheduler delay (actual_dispatch - ready_time)",
                        "Ready Time": "Parent requests completion + wait_after_ready (if available)",
                    },
                },
                {
                    "title": "Deviation Statistics (seconds)",
                    "results": stats,
                },
                {
                    "title": "Violation Info",
                    "results": {
                        "Late Threshold": f"{late_threshold}s (dispatched >{late_threshold}s after ready)",
                        "Violations": str(len(violations)),
                    },
                },
            ],
        }

        if violations:
            summary["sections"].append(
                {
                    "title": "Violation Details (first 5)",
                    "results": {
                        f"#{i+1}": f"session={v['session_id']} (size={v.get('session_size','?')}), diff={v['diff']:.4f}s"
                        for i, v in enumerate(violations[:5])
                    },
                }
            )

        return TestResult(summary=summary, passed=passed)

    def check_session_concurrency(self) -> TestResult:
        """
        Verify that the number of concurrent sessions tracks the target,
        respecting the ramp-up period.

        A violation occurs when concurrency exceeds the expected limit at any point.
        During ramp-up: limit is linearly interpolated from 0 to target.
        After ramp-up (steady-state): limit is target_concurrent_sessions.
        """
        session_col = self._get_col("session_id")
        dispatched_col = self._get_col("scheduler_dispatched_at")

        # Get benchmark end time (last request completion)
        all_ends = self.merged_df[dispatched_col] + self.merged_df["end_to_end_latency"]
        benchmark_end_time = all_ends.max()

        intervals = []
        for session_id, group in self.merged_df.groupby(session_col):
            starts = group[dispatched_col]
            ends = group[dispatched_col] + group["end_to_end_latency"]

            s_start = starts.min()

            # Check if all requests in session were dispatched
            # session_size tells us expected total requests
            session_size_col = self._get_col("session_size")
            if session_size_col in group.columns:
                expected_requests = group[session_size_col].iloc[0]
                actual_requests = len(group)

                if actual_requests < expected_requests:
                    # Session is still active (waiting for more requests)
                    # Consider it active until benchmark end
                    s_end = benchmark_end_time
                else:
                    # All requests dispatched, use last completion time
                    s_end = ends.max()
            else:
                # No session_size info, fall back to request-based measurement
                s_end = ends.max()

            intervals.append((s_start, s_end))

        # Sweep line to track concurrency over time
        events = []
        for s, e in intervals:
            events.append((s, 1))
            events.append((e, -1))

        events.sort()

        target = self.config.traffic_scheduler.target_concurrent_sessions  # type: ignore
        rampup_seconds = float(self.config.traffic_scheduler.rampup_seconds)  # type: ignore

        current_concurrency = 0
        ramp_violations = []
        steady_violations = []
        time_series = []  # [(time, concurrency)]

        for t, change in events:
            current_concurrency += change
            time_series.append((t, current_concurrency))

            # Compute limit at this time
            if rampup_seconds > 0 and t < rampup_seconds:
                limit = int(target * (t / rampup_seconds))
                phase_violations = ramp_violations
            else:
                limit = target
                phase_violations = steady_violations

            if current_concurrency > limit:
                phase_violations.append(
                    {"time": t, "concurrency": current_concurrency, "limit": limit}
                )

        violations = ramp_violations + steady_violations

        if not time_series:
            return TestResult(
                summary={
                    "name": "Session Concurrency Check",
                    "sections": [
                        {"title": "No Data", "results": {"Status": "No sessions found"}}
                    ],
                },
                passed=True,
            )

        # Compute detailed statistics
        benchmark_start = time_series[0][0]
        benchmark_end = time_series[-1][0]
        total_duration = benchmark_end - benchmark_start

        # Helper to generate smoothed time series
        def smooth_time_series(
            series: List[Tuple[float, int]], target_val: int, tolerance: float
        ) -> List[Tuple[float, int]]:
            """Return a new time series with small dips below target filled."""
            if not series:
                return []

            smoothed = []

            # We iterate intervals. valid interval i is from series[i].t to series[i+1].t
            # series[i].c is the value.
            for i in range(len(series) - 1):
                t_curr, c_curr = series[i]
                t_next, _ = series[i + 1]
                duration = t_next - t_curr

                # Turnaround "gap" usually: c checks below target, duration short.
                # If short duration and below target, bridge it by assuming target.
                # "discounting gaps of a particular size".
                if c_curr < target_val and duration <= tolerance and duration > 0:
                    c_curr = max(c_curr, target_val)

                smoothed.append((t_curr, c_curr))

            # append valid last point
            smoothed.append(series[-1])
            return smoothed

        # Helper to compute time-weighted stats over an interval
        def compute_phase_stats(
            series: List[Tuple[float, int]],
            phase_start: float,
            phase_end: float,
            limit_fn: Callable[[float], float],
        ):
            """Compute time-weighted concurrency stats for a phase."""
            time_below = 0.0
            time_at = 0.0
            time_above = 0.0
            weighted_sum = 0.0
            min_concurrency = float("inf")
            max_concurrency = 0
            duration_by_concurrency: Dict[int, float] = {}

            prev_time = phase_start
            prev_concurrency = 0

            for t, c in series:
                if t <= phase_start:
                    prev_concurrency = c
                    continue
                if t > phase_end:
                    break

                # Account for time from prev_time to min(t, phase_end)
                interval_start = max(prev_time, phase_start)
                interval_end = min(t, phase_end)
                dt = interval_end - interval_start

                if dt > 0:
                    weighted_sum += prev_concurrency * dt
                    min_concurrency = min(min_concurrency, prev_concurrency)
                    max_concurrency = max(max_concurrency, prev_concurrency)
                    duration_by_concurrency[prev_concurrency] = (
                        duration_by_concurrency.get(prev_concurrency, 0.0) + dt
                    )

                    expected_limit = limit_fn((interval_start + interval_end) * 0.5)
                    if prev_concurrency < expected_limit:
                        time_below += dt
                    elif prev_concurrency > expected_limit:
                        time_above += dt
                    else:
                        time_at += dt

                prev_time = t
                prev_concurrency = c

            # Handle final segment to phase_end
            if prev_time < phase_end:
                dt = phase_end - max(prev_time, phase_start)
                if dt > 0:
                    weighted_sum += prev_concurrency * dt
                    min_concurrency = min(min_concurrency, prev_concurrency)
                    max_concurrency = max(max_concurrency, prev_concurrency)
                    duration_by_concurrency[prev_concurrency] = (
                        duration_by_concurrency.get(prev_concurrency, 0.0) + dt
                    )

                    interval_start = max(prev_time, phase_start)
                    interval_end = phase_end
                    expected_limit = limit_fn((interval_start + interval_end) * 0.5)

                    if prev_concurrency < expected_limit:
                        time_below += dt
                    elif prev_concurrency > expected_limit:
                        time_above += dt
                    else:
                        time_at += dt

            phase_duration = phase_end - phase_start
            mean_concurrency = (
                weighted_sum / phase_duration if phase_duration > 0 else 0
            )
            median_concurrency = 0.0
            if phase_duration > 0 and duration_by_concurrency:
                midpoint = phase_duration / 2
                cumulative = 0.0
                for concurrency in sorted(duration_by_concurrency):
                    cumulative += duration_by_concurrency[concurrency]
                    if cumulative >= midpoint:
                        median_concurrency = float(concurrency)
                        break

            return {
                "duration": phase_duration,
                "time_below": time_below,
                "time_at": time_at,
                "time_above": time_above,
                "min": min_concurrency if min_concurrency != float("inf") else 0,
                "max": max_concurrency,
                "mean": mean_concurrency,
                "median": median_concurrency,
            }

        max_observed = max(c for _, c in time_series)
        runtime_timeout = 0.0
        runtime_cfg = getattr(self.config, "runtime", None)
        if runtime_cfg is not None:
            runtime_timeout = float(
                getattr(runtime_cfg, "benchmark_timeout", 0.0) or 0.0
            )
        timeout_cutoff = None
        timeout_triggered = False
        if runtime_timeout > 0:
            timeout_cutoff = benchmark_start + runtime_timeout
            timeout_triggered = benchmark_end >= timeout_cutoff
        post_timeout_context = (
            "After timeout we keep \ndispatching sessions until requests dispatched before \ntimeout finish, but the dispatch "
            "trace stops \nrecording them. Concurrency can look artificially low, so the \n"
            "post-timeout phase isolates this period."
        )

        sections = [
            {
                "title": "Configuration",
                "results": {
                    "Target Concurrency": str(target),
                    "Ramp-Up Period": f"{rampup_seconds:.1f}s",
                    "Benchmark Timeout": (
                        f"{runtime_timeout:.1f}s" if runtime_timeout > 0 else "Disabled"
                    ),
                },
            }
        ]

        def steady_limit_fn(_: float) -> float:
            return target

        def rampup_limit_fn(t: float) -> float:
            if rampup_seconds <= 0:
                return target

            relative_t = t - benchmark_start
            clamped = min(max(relative_t, 0.0), rampup_seconds)
            return int(target * (clamped / rampup_seconds))

        # Ramp-up phase analysis
        if rampup_seconds > 0:
            rampup_end = min(benchmark_start + rampup_seconds, benchmark_end)
            rampup_stats = compute_phase_stats(
                time_series, benchmark_start, rampup_end, rampup_limit_fn
            )
            rampup_duration = rampup_stats["duration"]

            if rampup_duration > 0:
                pct_below = 100.0 * rampup_stats["time_below"] / rampup_duration
                pct_at_or_above = (
                    100.0
                    * (rampup_stats["time_at"] + rampup_stats["time_above"])
                    / rampup_duration
                )

                sections.append(
                    {
                        "title": f"Ramp-Up Phase (0s to {rampup_seconds:.1f}s)",
                        "results": {
                            "Actual Duration": f"{rampup_duration:.2f}s",
                            "Time Below Expected": f"{rampup_stats['time_below']:.2f}s ({pct_below:.1f}%)",
                            "Time At/Above Expected": f"{rampup_stats['time_at'] + rampup_stats['time_above']:.2f}s ({pct_at_or_above:.1f}%)",
                            "Max Observed": str(rampup_stats["max"]),
                        },
                    }
                )

        # Steady-state phase analysis
        steady_start = (
            benchmark_start + rampup_seconds if rampup_seconds > 0 else benchmark_start
        )
        if steady_start < benchmark_end:
            stable_end = benchmark_end
            wind_down_reason = None

            if timeout_triggered and timeout_cutoff is not None:
                stable_end = min(stable_end, timeout_cutoff)
                wind_down_reason = f"Benchmark timeout reached ({runtime_timeout:.2f}s)"
            else:
                threshold = int(target * 0.9)
                # heuristic only used when the timeout-based cutoff is not applicable.
                for t, c in reversed(time_series):
                    if t <= steady_start:
                        break
                    if c >= threshold:
                        stable_end = t
                        break

            if timeout_triggered:
                sections.append(
                    {
                        "title": "Post-Timeout Context",
                        "results": {"Explanation": post_timeout_context},
                    }
                )

            # Compute stats for full steady-state phase (including wind-down)
            full_stats = compute_phase_stats(
                time_series, steady_start, benchmark_end, steady_limit_fn
            )
            full_duration = full_stats["duration"]

            # Compute stats for stable period only (excluding wind-down)
            stable_duration = stable_end - steady_start
            stable_stats = None
            wind_down_duration = max(benchmark_end - stable_end, 0.0)
            if stable_duration > 0 and stable_end < benchmark_end:
                stable_stats = compute_phase_stats(
                    time_series, steady_start, stable_end, steady_limit_fn
                )
            else:
                wind_down_duration = 0.0

            if full_duration > 0:
                pct_below = 100.0 * full_stats["time_below"] / full_duration
                pct_at = 100.0 * full_stats["time_at"] / full_duration
                pct_above = 100.0 * full_stats["time_above"] / full_duration

                sections.append(
                    {
                        "title": f"Steady-State Phase - Full ({rampup_seconds:.1f}s to end)",
                        "results": {
                            "Duration": f"{full_duration:.2f}s",
                            "Time Below Target": f"{full_stats['time_below']:.2f}s ({pct_below:.1f}%)",
                            "Time At Target": f"{full_stats['time_at']:.2f}s ({pct_at:.1f}%)",
                            "Time Above Target": f"{full_stats['time_above']:.2f}s ({pct_above:.1f}%)",
                            "Min Observed": str(full_stats["min"]),
                            "Mean Observed": f"{full_stats['mean']:.2f}",
                            "Median Observed": f"{full_stats['median']:.2f}",
                            "Max Observed": str(full_stats["max"]),
                        },
                    }
                )

            if stable_stats and stable_stats["duration"] > 0:
                s_duration = stable_stats["duration"]
                s_pct_below = 100.0 * stable_stats["time_below"] / s_duration
                s_pct_at = 100.0 * stable_stats["time_at"] / s_duration
                s_pct_above = 100.0 * stable_stats["time_above"] / s_duration

                stable_results = {
                    "Duration": f"{s_duration:.2f}s",
                    "Time Below Target": f"{stable_stats['time_below']:.2f}s ({s_pct_below:.1f}%)",
                    "Time At Target": f"{stable_stats['time_at']:.2f}s ({s_pct_at:.1f}%)",
                    "Time Above Target": f"{stable_stats['time_above']:.2f}s ({s_pct_above:.1f}%)",
                    "Min Observed": str(stable_stats["min"]),
                    "Mean Observed": f"{stable_stats['mean']:.2f}",
                    "Median Observed": f"{stable_stats['median']:.2f}",
                    "Max Observed": str(stable_stats["max"]),
                }
                if wind_down_reason and timeout_triggered:
                    stable_results["Post-Timeout Trigger"] = wind_down_reason

                stable_title_suffix = (
                    f"excludes {wind_down_duration:.1f}s post-timeout"
                    if timeout_triggered
                    else f"excludes {wind_down_duration:.1f}s tail"
                )
                sections.append(
                    {
                        "title": f"Steady-State Phase - Stable ({stable_title_suffix})",
                        "results": stable_results,
                    }
                )

                # Smoothed stable stats
                gap_tolerance = 0.5
                smoothed_series = smooth_time_series(time_series, target, gap_tolerance)
                smoothed_stats = compute_phase_stats(
                    smoothed_series, steady_start, stable_end, steady_limit_fn
                )

                sm_duration = smoothed_stats["duration"]
                if sm_duration > 0:
                    sm_pct_below = 100.0 * smoothed_stats["time_below"] / sm_duration
                    sm_pct_at = 100.0 * smoothed_stats["time_at"] / sm_duration
                    sm_pct_above = 100.0 * smoothed_stats["time_above"] / sm_duration

                    smoothed_title = (
                        f"Steady-State Phase - Stable Smoothed (pre-timeout, ignoring gaps <= {gap_tolerance}s)"
                        if timeout_triggered
                        else f"Steady-State Phase - Stable Smoothed (ignoring gaps <= {gap_tolerance}s)"
                    )
                    smoothed_results = {
                        "Duration": f"{sm_duration:.2f}s",
                        "Time Below Target": f"{smoothed_stats['time_below']:.2f}s ({sm_pct_below:.1f}%)",
                        "Time At Target": f"{smoothed_stats['time_at']:.2f}s ({sm_pct_at:.1f}%)",
                        "Time Above Target": f"{smoothed_stats['time_above']:.2f}s ({sm_pct_above:.1f}%)",
                        "Min Observed": str(smoothed_stats["min"]),
                        "Mean Observed": f"{smoothed_stats['mean']:.2f}",
                        "Median Observed": f"{smoothed_stats['median']:.2f}",
                        "Max Observed": str(smoothed_stats["max"]),
                    }
                    if timeout_triggered:
                        smoothed_results["Post-Timeout Coverage"] = (
                            f"Excluded (timeout at {runtime_timeout:.2f}s)"
                        )
                        if wind_down_reason:
                            smoothed_results["Post-Timeout Trigger"] = wind_down_reason

                    sections.append(
                        {
                            "title": smoothed_title,
                            "results": smoothed_results,
                        }
                    )

                if timeout_triggered and wind_down_duration > 0:
                    wind_down_stats = compute_phase_stats(
                        time_series, stable_end, benchmark_end, steady_limit_fn
                    )
                    wd_duration = wind_down_stats["duration"]
                    if wd_duration > 0:
                        wd_pct_below = (
                            100.0 * wind_down_stats["time_below"] / wd_duration
                        )
                        wd_pct_at = 100.0 * wind_down_stats["time_at"] / wd_duration
                        wd_pct_above = (
                            100.0 * wind_down_stats["time_above"] / wd_duration
                        )

                        wind_down_results = {
                            "Duration": f"{wd_duration:.2f}s",
                            "Time Below Target": f"{wind_down_stats['time_below']:.2f}s ({wd_pct_below:.1f}%)",
                            "Time At Target": f"{wind_down_stats['time_at']:.2f}s ({wd_pct_at:.1f}%)",
                            "Time Above Target": f"{wind_down_stats['time_above']:.2f}s ({wd_pct_above:.1f}%)",
                            "Min Observed": str(wind_down_stats["min"]),
                            "Mean Observed": f"{wind_down_stats['mean']:.2f}",
                            "Median Observed": f"{wind_down_stats['median']:.2f}",
                            "Max Observed": str(wind_down_stats["max"]),
                            "Explanation": post_timeout_context,
                        }
                        if wind_down_reason:
                            wind_down_results["Trigger"] = wind_down_reason

                        sections.append(
                            {
                                "title": "Post-Timeout Phase",
                                "results": wind_down_results,
                            }
                        )

        # Overall statistics
        sections.append(
            {
                "title": "Overall Statistics",
                "results": {
                    "Total Duration": f"{total_duration:.2f}s",
                    "Max Observed Concurrency": str(max_observed),
                },
            }
        )

        # Violation info
        sections.append(
            {
                "title": "Violation Info",
                "results": {
                    "Definition": "Concurrency exceeds phase-specific limit at any point",
                    "Ramp-Up Limit": f"Linear from 0 to {target}",
                    "Steady-State Limit": str(target),
                    "Ramp-Up Violations": str(len(ramp_violations)),
                    "Steady-State Violations": str(len(steady_violations)),
                    "Total Violations": str(len(violations)),
                },
            }
        )

        def append_violation_sections(
            title: str, violation_list: List[Dict[str, float]]
        ):
            if not violation_list:
                return

            overshoots = np.array(
                [v["concurrency"] - v["limit"] for v in violation_list]
            )
            sections.append(
                {
                    "title": f"{title} Violation Distribution (overshoot = concurrency - limit)",
                    "results": {
                        "Count": str(len(violation_list)),
                        "Min Overshoot": str(int(np.min(overshoots))),
                        "Mean Overshoot": f"{np.mean(overshoots):.2f}",
                        "Median Overshoot": f"{np.median(overshoots):.1f}",
                        "P95 Overshoot": f"{np.percentile(overshoots, 95):.1f}",
                        "Max Overshoot": str(int(np.max(overshoots))),
                    },
                }
            )

            violation_times = np.array([v["time"] for v in violation_list])
            sections.append(
                {
                    "title": f"{title} Violation Time Distribution (seconds)",
                    "results": {
                        "First Violation": f"{np.min(violation_times):.2f}s",
                        "Last Violation": f"{np.max(violation_times):.2f}s",
                        "Median Time": f"{np.median(violation_times):.2f}s",
                    },
                }
            )

            sections.append(
                {
                    "title": f"{title} Violation Details (first 5)",
                    "results": {
                        f"#{i+1}": f"t={v['time']:.2f}s, concurrency={v['concurrency']}, limit={v['limit']}"
                        for i, v in enumerate(violation_list[:5])
                    },
                }
            )

        append_violation_sections("Ramp-Up", ramp_violations)
        append_violation_sections("Steady-State", steady_violations)

        passed = len(violations) == 0

        summary = {"name": "Session Concurrency Check", "sections": sections}
        return TestResult(summary=summary, passed=passed)

    def check_session_dispatch_rate(self) -> TestResult:
        """
        Verify that the actual session dispatch rate matches the expected arrival rate.

        Calculated as: (N - 1) / (last_start_time - first_start_time)
        Matches against the configured arrival_rate (for Poisson/Gamma) or 1/interval (for Fixed).
        """
        # Determine expected rate
        scheduler_config = self.config.traffic_scheduler
        interval_config = scheduler_config.interval_generator  # type: ignore
        generator_type = interval_config.get_type()

        expected_rate = 0.0
        if generator_type in [
            IntervalGeneratorType.POISSON,
            IntervalGeneratorType.GAMMA,
        ]:
            expected_rate = float(interval_config.arrival_rate)
        elif generator_type == IntervalGeneratorType.FIXED:
            interval = float(interval_config.interval)
            expected_rate = 1.0 / interval if interval > 0 else 0.0

        if expected_rate <= 0:
            return TestResult(
                summary={
                    "name": "Session Dispatch Rate Check",
                    "sections": [
                        {
                            "title": "Invalid Configuration",
                            "results": {
                                "Status": "Skipped",
                                "Reason": f"Expected rate <= 0 (type={generator_type.name})",
                            },
                        }
                    ],
                },
                passed=True,
            )

        # Get session start times
        session_col = self._get_col("session_id")
        dispatched_col = self._get_col("scheduler_dispatched_at")

        if self.merged_df.empty:
            return TestResult(
                summary={
                    "name": "Session Dispatch Rate Check",
                    "sections": [
                        {"title": "No Data", "results": {"Status": "No sessions found"}}
                    ],
                },
                passed=True,
            )

        # first request dispatch time for each session (min for multi root sessions)
        session_starts = self.merged_df.groupby(session_col)[dispatched_col].min()
        start_times = np.sort(np.array(session_starts.values))

        if len(start_times) < 2:
            return TestResult(
                summary={
                    "name": "Session Dispatch Rate Check",
                    "sections": [
                        {
                            "title": "Insufficient Data",
                            "results": {
                                "Status": "Skipped",
                                "Reason": f"Only {len(start_times)} session(s) found, need at least 2",
                            },
                        }
                    ],
                },
                passed=True,
            )

        duration = start_times[-1] - start_times[0]
        count = len(start_times) - 1
        actual_rate = count / duration if duration > 0 else 0.0

        inter_arrival_times = np.diff(start_times)
        if len(inter_arrival_times) == 0:
            pass

        error_pct = (
            abs(actual_rate - expected_rate) / expected_rate * 100.0
            if expected_rate > 0
            else 0.0
        )
        threshold_pct = 15.0

        passed = error_pct <= threshold_pct

        stats = {
            "Total Sessions": str(len(start_times)),
            "Measurement Duration": f"{duration:.4f}s",
            "Expected Rate": f"{expected_rate:.4f} sessions/sec",
            "Actual Rate": f"{actual_rate:.4f} sessions/sec",
            "Error": f"{error_pct:.2f}%",
            "Threshold": f"{threshold_pct:.1f}%",
        }

        inter_arrival_stats = {
            "Min": f"{np.min(inter_arrival_times):.4f}s",
            "Mean": f"{np.mean(inter_arrival_times):.4f}s",
            "Median": f"{np.median(inter_arrival_times):.4f}s",
            "P95": f"{np.percentile(inter_arrival_times, 95):.4f}s",
            "P99": f"{np.percentile(inter_arrival_times, 99):.4f}s",
            "Max": f"{np.max(inter_arrival_times):.4f}s",
            "Std Dev": f"{np.std(inter_arrival_times):.4f}s",
        }

        summary = {
            "name": "Session Dispatch Rate Check",
            "sections": [
                {
                    "title": "Rate Statistics",
                    "results": stats,
                },
                {
                    "title": "Inter-Arrival Time Statistics",
                    "results": inter_arrival_stats,
                },
            ],
        }

        if not passed:
            summary["sections"].append(
                {
                    "title": "Failure Reason",
                    "results": {
                        "Result": "FAILED",
                        "Explanation": f"Actual rate deviates from expected by > {threshold_pct}%",
                    },
                }
            )

        return TestResult(summary=summary, passed=passed)

    def check_prompt_length(self) -> TestResult:
        """
        Verify that the generated prompt length matches the target prompt length.
        Checks deviation: num_delta_prompt_tokens - target_num_delta_prompt_tokens.
        """
        target_col = self._get_col("target_num_delta_prompt_tokens")
        actual_col_delta = self._get_col("num_delta_prompt_tokens")

        # Fallback to checking num_prompt_tokens if delta not available (legacy)
        # But user specifically asked for delta. The generators populate "target_prompt_tokens".
        # Evaluator maps this to "target_num_delta_prompt_tokens".
        # "num_delta_prompt_tokens" is from channel metrics.

        if (
            target_col not in self.merged_df.columns
            or actual_col_delta not in self.merged_df.columns
        ):
            return TestResult(
                summary={
                    "name": "Prompt Length Check",
                    "sections": [
                        {
                            "title": "Missing Data",
                            "results": {
                                "Status": "Skipped",
                                "Reason": f"Columns '{target_col}' or '{actual_col_delta}' missing.",
                            },
                        }
                    ],
                },
                passed=True,
            )

        # Filter rows where target is present (non-null and > 0)
        df = self.merged_df.dropna(subset=[target_col, actual_col_delta])
        if df.empty:
            return TestResult(
                summary={
                    "name": "Prompt Length Check",
                    "sections": [
                        {
                            "title": "No Target Data",
                            "results": {
                                "Status": "Skipped",
                                "Reason": "No requests found with target prompt tokens specified.",
                            },
                        }
                    ],
                },
                passed=True,
            )

        df["prompt_len_diff"] = df[actual_col_delta] - df[target_col]
        deviations = df["prompt_len_diff"].to_numpy()

        # Relaxed check with threshold
        threshold = 15.0
        violations = df[df["prompt_len_diff"].abs() > threshold]
        violation_count = len(violations)
        total_count = len(df)

        passed = violation_count == 0

        stats = {
            "Total Requests Checked": str(total_count),
            "Exact Matches": f"{len(df[df['prompt_len_diff'] == 0])} ({100.0 * len(df[df['prompt_len_diff'] == 0]) / total_count:.1f}%)",
            "Mismatches (All)": f"{len(df[df['prompt_len_diff'] != 0])} ({100.0 * len(df[df['prompt_len_diff'] != 0]) / total_count:.1f}%)",
            "Violations (> +/-15)": f"{violation_count} ({100.0 * violation_count / total_count:.1f}%)",
        }

        if deviations.size > 0:
            stats.update(
                {
                    "Min Deviation": f"{np.min(deviations):.1f}",
                    "Mean Deviation": f"{np.mean(deviations):.2f}",
                    "Median Deviation": f"{np.median(deviations):.1f}",
                    "P95 Deviation": f"{np.percentile(deviations, 95):.1f}",
                    "P99 Deviation": f"{np.percentile(deviations, 99):.1f}",
                    "Max Deviation": f"{np.max(deviations):.1f}",
                    "Std Dev": f"{np.std(deviations):.2f}",
                }
            )

        summary = {
            "name": "Prompt Length Check",
            "sections": [
                {
                    "title": "Description",
                    "results": {
                        "Metric": "Prompt Length Deviation (Actual - Target)",
                        "Target": "Specified target_prompt_tokens",
                        "Threshold": f"<= +/- {threshold}",
                    },
                },
                {
                    "title": "Statistics",
                    "results": stats,
                },
            ],
        }

        if violation_count > 0:
            # Add top violations (largest absolute difference)
            violations_df = cast(pd.DataFrame, violations.copy())
            violations_df["abs_diff"] = np.abs(violations_df["prompt_len_diff"])
            top_violations = violations_df.nlargest(5, "abs_diff")

            summary["sections"].append(
                {
                    "title": "Top Violations (Largest Absolute Difference)",
                    "results": {
                        f"#{i+1}": f"req={row['request_id']}, target={row[target_col]}, actual={row[actual_col_delta]}, diff={row['prompt_len_diff']}"
                        for i, (_, row) in enumerate(top_violations.iloc[:5].iterrows())
                    },
                }
            )

        return TestResult(summary=summary, passed=passed)

    def check_output_length(self) -> TestResult:
        """
        Verify that the generated output length matches the requested output length.
        Checks deviation: num_output_tokens - num_requested_output_tokens.
        """
        target_col = self._get_col("num_requested_output_tokens")
        actual_col = self._get_col("num_output_tokens")

        if (
            target_col not in self.merged_df.columns
            or actual_col not in self.merged_df.columns
        ):
            return TestResult(
                summary={
                    "name": "Output Length Check",
                    "sections": [
                        {
                            "title": "Missing Data",
                            "results": {
                                "Status": "Skipped",
                                "Reason": f"Columns '{target_col}' or '{actual_col}' missing.",
                            },
                        }
                    ],
                },
                passed=True,
            )

        # Filter rows where target is present (non-null and > 0)
        df = self.merged_df.dropna(subset=[target_col, actual_col])
        if df.empty:
            return TestResult(
                summary={
                    "name": "Output Length Check",
                    "sections": [
                        {
                            "title": "No Data",
                            "results": {
                                "Status": "Skipped",
                                "Reason": "No requests found with requested output tokens specified.",
                            },
                        }
                    ],
                },
                passed=True,
            )

        df["output_len_diff"] = df[actual_col] - df[target_col]
        deviations = df["output_len_diff"].to_numpy()

        # Relaxed check with threshold
        threshold = 15.0  # Allow some deviation
        violations = df[df["output_len_diff"].abs() > threshold]
        violation_count = len(violations)
        total_count = len(df)

        passed = violation_count == 0

        stats = {
            "Total Requests Checked": str(total_count),
            "Exact Matches": f"{len(df[df['output_len_diff'] == 0])} ({100.0 * len(df[df['output_len_diff'] == 0]) / total_count:.1f}%)",
            "Mismatches (All)": f"{len(df[df['output_len_diff'] != 0])} ({100.0 * len(df[df['output_len_diff'] != 0]) / total_count:.1f}%)",
            "Violations (> +/-15)": f"{violation_count} ({100.0 * violation_count / total_count:.1f}%)",
        }

        if deviations.size > 0:
            stats.update(
                {
                    "Min Deviation": f"{np.min(deviations):.1f}",
                    "Mean Deviation": f"{np.mean(deviations):.2f}",
                    "Median Deviation": f"{np.median(deviations):.1f}",
                    "P95 Deviation": f"{np.percentile(deviations, 95):.1f}",
                    "P99 Deviation": f"{np.percentile(deviations, 99):.1f}",
                    "Max Deviation": f"{np.max(deviations):.1f}",
                    "Std Dev": f"{np.std(deviations):.2f}",
                }
            )

        summary = {
            "name": "Output Length Check",
            "sections": [
                {
                    "title": "Description",
                    "results": {
                        "Metric": "Output Length Deviation (Actual - Requested)",
                        "Target": "num_requested_output_tokens",
                        "Threshold": f"<= +/- {threshold}",
                    },
                },
                {
                    "title": "Statistics",
                    "results": stats,
                },
            ],
        }

        if violation_count > 0:
            # Add top violations (largest absolute difference)
            violations_df = cast(pd.DataFrame, violations.copy())
            violations_df["abs_diff"] = np.abs(violations_df["output_len_diff"])
            top_violations = violations_df.nlargest(5, "abs_diff")

            summary["sections"].append(
                {
                    "title": "Top Violations (Largest Absolute Difference)",
                    "results": {
                        f"#{i+1}": f"req={row['request_id']}, target={row[target_col]}, actual={row[actual_col]}, diff={row['output_len_diff']}"
                        for i, (_, row) in enumerate(top_violations.iloc[:5].iterrows())
                    },
                }
            )

        return TestResult(summary=summary, passed=passed)

    def check_lifecycle_timing_delays(self) -> TestResult:
        """
        Report statistics on lifecycle timing delays between pipeline stages.

        Computes percentiles for:
        - scheduler_ready_at -> scheduler_dispatched_at (dispatch delay)
        - scheduler_dispatched_at -> client_picked_up_at (queue wait time)
        - client_completed_at -> result_processed_at (result processing delay)
        """

        def compute_delay_stats(delays: np.ndarray, name: str) -> Dict[str, str]:
            """Compute percentile statistics for a delay array."""
            if delays.size == 0:
                return {f"{name} (no data)": "N/A"}

            # Filter out NaN values
            valid_delays = delays[~np.isnan(delays)]
            if valid_delays.size == 0:
                return {f"{name} (no valid data)": "N/A"}

            return {
                "Count": str(valid_delays.size),
                "Min": f"{np.min(valid_delays):.4f}s",
                "Mean": f"{np.mean(valid_delays):.4f}s",
                "Median": f"{np.median(valid_delays):.4f}s",
                "P95": f"{np.percentile(valid_delays, 95):.4f}s",
                "P99": f"{np.percentile(valid_delays, 99):.4f}s",
                "Max": f"{np.max(valid_delays):.4f}s",
                "Std Dev": f"{np.std(valid_delays):.4f}s",
            }

        sections = []

        # Check if lifecycle columns exist
        ready_col = "scheduler_ready_at"
        dispatched_col = "scheduler_dispatched_at"
        pickup_col = "client_picked_up_at"
        completed_col = "client_completed_at"
        processed_col = "result_processed_at"

        missing_cols = []
        for col in [
            ready_col,
            dispatched_col,
            pickup_col,
            completed_col,
            processed_col,
        ]:
            if col not in self.metrics_df.columns:
                missing_cols.append(col)

        if missing_cols:
            return TestResult(
                summary={
                    "name": "Lifecycle Timing Delays Check",
                    "sections": [
                        {
                            "title": "Missing Data",
                            "results": {
                                "Missing Columns": ", ".join(missing_cols),
                                "Resolution": "Re-run benchmark with updated code that captures lifecycle timestamps.",
                            },
                        }
                    ],
                },
                passed=True,  # Not a failure, just missing data
            )

        df = self.metrics_df.copy()

        # 1. Ready-to-Dispatch delay
        ready_to_dispatch = (df[dispatched_col] - df[ready_col]).values
        sections.append(
            {
                "title": "Ready-to-Dispatch Delay (scheduler_dispatched_at - scheduler_ready_at)",
                "results": compute_delay_stats(ready_to_dispatch, "Delay"),
            }
        )

        # 2. Dispatch-to-Pickup delay (queue wait time)
        dispatch_to_pickup = (df[pickup_col] - df[dispatched_col]).values
        sections.append(
            {
                "title": "Dispatch-to-Pickup Delay (client_picked_up_at - scheduler_dispatched_at)",
                "results": compute_delay_stats(dispatch_to_pickup, "Delay"),
            }
        )

        # 3. Completion-to-Result-Processing delay
        completion_to_processing = (df[processed_col] - df[completed_col]).values
        sections.append(
            {
                "title": "Completion-to-Result-Processing Delay (result_processed_at - client_completed_at)",
                "results": compute_delay_stats(completion_to_processing, "Delay"),
            }
        )

        return TestResult(
            summary={
                "name": "Lifecycle Timing Delays Check",
                "sections": sections,
            },
            passed=True,  # No pass/fail criteria for now
        )
