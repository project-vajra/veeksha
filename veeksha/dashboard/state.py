import threading
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Deque, Dict, List, Optional, Union

from veeksha.dashboard.events import (
    BenchmarkStatusEvent,
    CapacitySearchEvent,
    DashboardEvent,
    RequestCompletedEvent,
    RequestErrorEvent,
    RequestStartedEvent,
    TokenBatchEvent,
)


@dataclass
class LiveRequestInfo:
    request_id: Union[str, int]  # Can be string or int depending on source
    start_timestamp: float
    input_tokens: int
    current_output_tokens: int = 0
    ttft_ms: Optional[float] = None
    current_tpot_ms: Optional[float] = None
    progress_pct: float = 0.0
    is_waiting_first_token: bool = True


@dataclass
class AggregateStats:
    completed_count: int = 0
    error_count: int = 0
    total_requests: int = 0

    recent_ttft_ms: Deque[float] = field(default_factory=lambda: deque())
    recent_tpot_ms: Deque[float] = field(default_factory=lambda: deque())
    recent_tbt_ms: Deque[float] = field(default_factory=lambda: deque())
    recent_latency_ms: Deque[float] = field(default_factory=lambda: deque())

    @property
    def avg_ttft_ms(self) -> float:
        return mean(self.recent_ttft_ms) if self.recent_ttft_ms else 0.0

    @property
    def avg_tpot_ms(self) -> float:
        return mean(self.recent_tpot_ms) if self.recent_tpot_ms else 0.0

    @property
    def avg_tbt_ms(self) -> float:
        return mean(self.recent_tbt_ms) if self.recent_tbt_ms else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return mean(self.recent_latency_ms) if self.recent_latency_ms else 0.0


@dataclass
class CapacitySearchState:
    current_qps: float = 0.0
    current_iteration: int = 0
    total_iterations: int = 0
    search_left: float = 0.0
    search_right: float = 0.0
    is_under_sla: bool = False
    slo_target: str = ""
    slo_metrics: Dict[str, float] = field(default_factory=dict)
    qps_history: List[Dict[str, any]] = field(
        default_factory=list
    )  # list of dicts with qps, under_sla, slo_metrics, from_cache
    best_qps: Optional[float] = None
    best_slo_metrics: Optional[Dict[str, float]] = None
    is_active: bool = False  # Whether capacity search is currently running
    is_complete: bool = False
    current_from_cache: bool = False  # Whether current iteration used cached results


@dataclass
class SingleBenchmarkState:
    """State for a single benchmark run"""

    benchmark_id: str
    live_requests: Dict[str, LiveRequestInfo] = field(default_factory=dict)
    completed_requests: Dict[str, LiveRequestInfo] = field(default_factory=dict)
    aggregate_stats: AggregateStats = field(default_factory=AggregateStats)
    capacity_search: CapacitySearchState = field(default_factory=CapacitySearchState)
    benchmark_start_time: Optional[float] = None
    benchmark_end_time: Optional[float] = None
    current_qps: float = 0.0


class DashboardState:
    def __init__(
        self, max_live_requests: int = 50, chart_window_seconds: Optional[float] = None
    ):
        self._lock = threading.RLock()
        self.max_live_requests = max_live_requests
        self.chart_window_seconds = (
            chart_window_seconds  # Optional time window for charts
        )

        # Track multiple benchmarks by ID
        self.benchmarks: Dict[str, SingleBenchmarkState] = {}
        self.active_benchmark_id: str = "default"  # Currently selected benchmark

    def _get_or_create_benchmark(self, benchmark_id: str) -> SingleBenchmarkState:
        """Get existing benchmark state or create a new one"""
        if benchmark_id not in self.benchmarks:
            self.benchmarks[benchmark_id] = SingleBenchmarkState(
                benchmark_id=benchmark_id
            )
            # Set as active if it's the first benchmark
            if len(self.benchmarks) == 1:
                self.active_benchmark_id = benchmark_id
        return self.benchmarks[benchmark_id]

    def apply(self, event: DashboardEvent) -> None:
        """Apply an event to update dashboard state with locking for thread safety"""
        with self._lock:
            match event:
                case RequestStartedEvent():
                    self._handle_request_started(event)
                case TokenBatchEvent():
                    self._handle_token_batch(event)
                case RequestCompletedEvent():
                    self._handle_request_completed(event)
                case CapacitySearchEvent():
                    self._handle_capacity_search(event)
                case BenchmarkStatusEvent():
                    self._handle_benchmark_status(event)
                case RequestErrorEvent():
                    self._handle_request_error(event)

    def _handle_request_started(self, event: RequestStartedEvent) -> None:
        benchmark = self._get_or_create_benchmark(event.benchmark_id)

        # Update start time, live requests, and increment total requests
        if benchmark.benchmark_start_time is None:
            benchmark.benchmark_start_time = event.timestamp

        if (
            len(benchmark.live_requests) >= self.max_live_requests
        ):  # throttle number of live requests
            oldest_id = min(
                benchmark.live_requests.keys(),
                key=lambda k: benchmark.live_requests[k].start_timestamp,
            )
            del benchmark.live_requests[oldest_id]

        # Normalize request_id to string for consistent dict lookups
        request_key = str(event.request_id)
        benchmark.live_requests[request_key] = LiveRequestInfo(
            request_id=event.request_id,
            start_timestamp=event.timestamp,
            input_tokens=event.input_tokens,
        )

        benchmark.aggregate_stats.total_requests += 1

    def _handle_token_batch(self, event: TokenBatchEvent) -> None:
        benchmark = self._get_or_create_benchmark(event.benchmark_id)

        # Update live request info
        request_key = str(event.request_id)
        if request_key in benchmark.live_requests:
            req = benchmark.live_requests[request_key]
            req.current_output_tokens = event.total_output_tokens
            req.ttft_ms = event.ttft_ms
            req.current_tpot_ms = event.current_tpot_ms
            req.is_waiting_first_token = (
                not event.is_first_token and req.ttft_ms is None
            )

        # Add live metrics to aggregate stats for graphing
        # Only add TTFT on first token to avoid duplicates
        if event.is_first_token and event.ttft_ms is not None and event.ttft_ms > 0:
            benchmark.aggregate_stats.recent_ttft_ms.append(event.ttft_ms)

        # For TPOT: we update continuously as it changes with more tokens
        # This gives live updates but will have duplicate request IDs - that's okay for visualization
        if event.current_tpot_ms is not None and event.current_tpot_ms > 0:
            benchmark.aggregate_stats.recent_tpot_ms.append(event.current_tpot_ms)

        # Add individual TBT values from this batch
        if event.recent_tbt_ms:
            for tbt in event.recent_tbt_ms:
                if tbt > 0:
                    benchmark.aggregate_stats.recent_tbt_ms.append(tbt)

    def _handle_request_completed(self, event: RequestCompletedEvent) -> None:
        benchmark = self._get_or_create_benchmark(event.benchmark_id)

        # Move from live requests to completed requests + update aggregate stats
        request_key = str(event.request_id)
        if request_key in benchmark.live_requests:
            # Move to completed requests with final state
            completed_req = benchmark.live_requests[request_key]
            completed_req.progress_pct = 100.0  # Mark as fully completed
            completed_req.is_waiting_first_token = False

            # Update with final metrics
            metrics = event.final_metrics
            if metrics.ttft > 0:
                completed_req.ttft_ms = metrics.ttft * 1000
            if metrics.tpot > 0:
                completed_req.current_tpot_ms = metrics.tpot * 1000

            # Move to completed collection
            benchmark.completed_requests[request_key] = completed_req
            del benchmark.live_requests[request_key]
        else:
            # Request was not in live_requests (throttled out), but still add to completed
            metrics = event.final_metrics
            completed_req = LiveRequestInfo(
                request_id=event.request_id,
                start_timestamp=event.timestamp,
                input_tokens=metrics.num_prompt_tokens,
                current_output_tokens=metrics.num_output_tokens,
                ttft_ms=metrics.ttft * 1000 if metrics.ttft > 0 else None,
                current_tpot_ms=metrics.tpot * 1000 if metrics.tpot > 0 else None,
                progress_pct=100.0,
                is_waiting_first_token=False,
            )
            benchmark.completed_requests[str(event.request_id)] = completed_req

        benchmark.aggregate_stats.completed_count += 1
        metrics = event.final_metrics

        # Note: TTFT, TPOT, and TBT are already added during TokenBatchEvent streaming
        # We only add end-to-end latency here since it's only available at completion
        if metrics.end_to_end_latency > 0:
            benchmark.aggregate_stats.recent_latency_ms.append(
                metrics.end_to_end_latency * 1000
            )

    def _handle_capacity_search(self, event: CapacitySearchEvent) -> None:
        benchmark = self._get_or_create_benchmark(event.benchmark_id)
        cs = benchmark.capacity_search

        # Mark capacity search as active on first event
        if not cs.is_active:
            cs.is_active = True

        cs.current_qps = event.current_qps
        cs.current_iteration = event.iteration
        cs.total_iterations = event.total_iterations
        cs.search_left = event.search_left
        cs.search_right = event.search_right
        cs.is_under_sla = event.is_under_sla
        cs.slo_target = event.slo_target
        cs.slo_metrics = event.slo_metrics.copy()
        cs.best_qps = event.best_qps
        cs.best_slo_metrics = (
            event.best_slo_metrics.copy() if event.best_slo_metrics else None
        )
        cs.is_complete = event.is_complete
        cs.current_from_cache = event.from_cache

        # Add to history (avoid duplicates by checking if QPS already exists)
        if not any(h.get("qps") == event.current_qps for h in cs.qps_history):
            cs.qps_history.append(
                {
                    "qps": event.current_qps,
                    "under_sla": event.is_under_sla,
                    "slo_metrics": event.slo_metrics.copy(),
                    "from_cache": event.from_cache,
                }
            )

    def _handle_benchmark_status(self, event: BenchmarkStatusEvent) -> None:
        benchmark = self._get_or_create_benchmark(event.benchmark_id)

        # Update current qps
        benchmark.current_qps = event.current_qps

        # Update aggregate stats from the event
        # This is important because it provides the final, authoritative counts
        # Even if individual request events were missed, this gives us the correct totals
        benchmark.aggregate_stats.total_requests = event.total_requests
        benchmark.aggregate_stats.completed_count = event.completed_requests
        benchmark.aggregate_stats.error_count = event.errored_requests

        # If all requests are completed or errored, mark benchmark as ended
        if (
            event.completed_requests + event.errored_requests == event.total_requests
            and event.total_requests > 0
            and benchmark.benchmark_end_time is None
        ):
            benchmark.benchmark_end_time = time.time()

    def _handle_request_error(self, event: RequestErrorEvent) -> None:
        benchmark = self._get_or_create_benchmark(event.benchmark_id)

        request_key = str(event.request_id)
        if request_key in benchmark.live_requests:
            del benchmark.live_requests[request_key]
        benchmark.aggregate_stats.error_count += 1

    # Getter methods for stats with locking for thread safety
    def get_benchmark_ids(self) -> List[str]:
        """Get list of all benchmark IDs"""
        with self._lock:
            return list(self.benchmarks.keys())

    def get_active_benchmark(self) -> Optional[SingleBenchmarkState]:
        """Get the currently active benchmark state"""
        with self._lock:
            return self.benchmarks.get(self.active_benchmark_id)

    def set_active_benchmark(self, benchmark_id: str) -> None:
        """Set the active benchmark by ID"""
        with self._lock:
            if benchmark_id in self.benchmarks:
                self.active_benchmark_id = benchmark_id

    def get_live_requests(
        self, benchmark_id: Optional[str] = None
    ) -> List[LiveRequestInfo]:
        """Get live requests for a specific benchmark or the active one"""
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                return list(self.benchmarks[bid].live_requests.values())
            return []

    def get_all_requests(
        self, benchmark_id: Optional[str] = None
    ) -> List[LiveRequestInfo]:
        """Get list of all requests (live + completed) for a specific benchmark"""
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                benchmark = self.benchmarks[bid]
                all_requests = []
                all_requests.extend(benchmark.live_requests.values())
                all_requests.extend(benchmark.completed_requests.values())
                return all_requests
            return []

    def get_completed_requests(
        self, benchmark_id: Optional[str] = None
    ) -> List[LiveRequestInfo]:
        """Get list of completed requests for a specific benchmark"""
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                return list(self.benchmarks[bid].completed_requests.values())
            return []

    def get_aggregate_stats(self, benchmark_id: Optional[str] = None) -> AggregateStats:
        """Get aggregate stats for a specific benchmark with thread-safe deque copies"""
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                stats = self.benchmarks[bid].aggregate_stats
                # Create a copy with snapshot of deques to avoid "deque mutated during iteration"
                return AggregateStats(
                    completed_count=stats.completed_count,
                    error_count=stats.error_count,
                    total_requests=stats.total_requests,
                    recent_ttft_ms=deque(list(stats.recent_ttft_ms), maxlen=100),
                    recent_tpot_ms=deque(list(stats.recent_tpot_ms), maxlen=100),
                    recent_tbt_ms=deque(list(stats.recent_tbt_ms), maxlen=100),
                    recent_latency_ms=deque(list(stats.recent_latency_ms), maxlen=100),
                )
            return AggregateStats()

    def get_aggregate_stats_snapshot(self, benchmark_id: Optional[str] = None) -> tuple:
        """Get a thread-safe snapshot of aggregate stats data

        Returns:
            Tuple of (completed_count, error_count, total_requests,
                     ttft_list, tpot_list, tbt_list, latency_list)
        """
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                stats = self.benchmarks[bid].aggregate_stats
                return (
                    stats.completed_count,
                    stats.error_count,
                    stats.total_requests,
                    list(stats.recent_ttft_ms),  # Create copies while holding lock
                    list(stats.recent_tpot_ms),
                    list(stats.recent_tbt_ms),
                    list(stats.recent_latency_ms),
                )
            return (0, 0, 0, [], [], [], [])

    def get_capacity_search_state(
        self, benchmark_id: Optional[str] = None
    ) -> CapacitySearchState:
        """Get capacity search state for a specific benchmark"""
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                return self.benchmarks[bid].capacity_search
            return CapacitySearchState()

    @property
    def capacity_search(self) -> CapacitySearchState:
        """Get capacity search state for the active benchmark (convenience property)"""
        return self.get_capacity_search_state()

    def get_benchmark_duration(self, benchmark_id: Optional[str] = None) -> float:
        """Get benchmark duration for a specific benchmark"""
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                benchmark = self.benchmarks[bid]
                if benchmark.benchmark_start_time is None:
                    return 0.0
                # If benchmark has ended, use end time; otherwise use current time
                end_time = benchmark.benchmark_end_time or time.time()
                return end_time - benchmark.benchmark_start_time
            return 0.0
