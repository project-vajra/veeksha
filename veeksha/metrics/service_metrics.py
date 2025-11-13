import threading
import time
from typing import Dict, Optional

from veeksha.config.metrics import MetricsConfig
from veeksha.logger import init_logger
from veeksha.metrics.metric_store import MetricStore
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


class ServiceMetrics:
    def __init__(
        self,
        timeout: float,
        max_requests: int,
        metrics_config: MetricsConfig,
    ) -> None:
        self.timeout = timeout
        self.max_requests = max_requests
        self.start_time = None
        self.end_time = None
        self.output_dir = metrics_config.output_dir
        self._stop_event = threading.Event()
        self._error_lock = threading.Lock()
        self._error_event = threading.Event()
        self._error: Optional[BaseException] = None

        # Counter for generated requests (protected by external generator_lock in prefetch threads)
        self.num_generated_requests = 0

        self.metric_store = MetricStore(
            timeout=timeout,
            max_requests=max_requests,
            metrics_config=metrics_config,
        )

        # synchronization for tail-phase state
        self._state_lock = threading.Lock()
        self._last_launched_request_id: Optional[int] = None  # as seen by dispatcher
        # tail-phase activation and data for first timeout crossing
        self._cutoff_active: bool = False
        self._cutoff_time: Optional[float] = None
        self._cutoff_last_launched_request_id: Optional[int] = None
        self._cutoff_launched_requests: int = 0
        # n outcomes (success or error) for pre-timeout launched requests
        self._num_pre_timeout_terminal: int = 0

    @property
    def num_requests(self) -> int:
        return self.metric_store.num_requests

    @property
    def num_completed_requests(self) -> int:
        return self.metric_store.num_completed_requests

    @property
    def num_errored_requests(self) -> int:
        return self.metric_store.num_errored_requests

    @property
    def duration(self):
        assert self.end_time is not None
        assert self.start_time is not None

        return self.end_time - self.start_time

    @property
    def completed_requests_per_min(self):
        return self.num_completed_requests / self.duration * 60

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    def should_stop(self):
        assert self.start_time is not None
        # never stop due to timeout if timeout is -1
        if self._stop_event.is_set() or self._error_event.is_set():
            return True
        if self.timeout == -1:
            return not (self.num_completed_requests < self.max_requests)

        now = time.perf_counter()
        elapsed = now - self.start_time

        # before cutoff we only stop if we have completed the max number of requests
        if not self._cutoff_active:
            if self.num_completed_requests >= self.max_requests:
                return True
            if elapsed < self.timeout:
                return False
            # tail-phase: timeout has been crossed
            with self._state_lock:
                if not self._cutoff_active:
                    self._cutoff_active = True
                    self._cutoff_time = now
                    self._cutoff_last_launched_request_id = (
                        self._last_launched_request_id
                    )
                    self._cutoff_launched_requests = min(
                        self.num_requests, self.max_requests
                    )
                    # count finished requests so far
                    initial_done = min(
                        self.num_completed_requests + self.num_errored_requests,
                        self._cutoff_launched_requests,
                    )
                    self._num_pre_timeout_terminal = initial_done
                    logger.info(
                        "Timeout reached; entering tail phase. Now waiting for %s requests to finish while still dispatching.",
                        self._cutoff_launched_requests - initial_done,
                    )

        # in tail phase we continue running until all pre-timout requests
        # are a success or error.
        # only requests launched before timeout count towards target
        with self._state_lock:
            target = self._cutoff_launched_requests
            done = self._num_pre_timeout_terminal
            if done >= target and self._cutoff_active:
                logger.info("Tail phase complete!")
        return done >= target

    def register_launched_request(self, request_id: Optional[int] = None):
        self.metric_store.register_launched_request()
        # track last launched request id for cutoff snapshot
        if request_id is not None:
            with self._state_lock:
                self._last_launched_request_id = request_id

    def add_request_metrics(self, request_metrics: RequestMetrics):
        # if this result is for a post-timeout request, we drop it
        is_filler = False
        req_id = request_metrics.request_id
        with self._state_lock:
            if (
                self._cutoff_active
                and self._cutoff_last_launched_request_id is not None
            ):
                if (
                    req_id is not None
                    and req_id > self._cutoff_last_launched_request_id
                ):
                    is_filler = True

        if not is_filler:
            self.metric_store.add_request_metrics(request_metrics)
        else:
            logger.debug(
                "Dropping metrics for filler request_id=%s (post-timeout).", str(req_id)
            )

        # In tail phase, count terminal outcomes for pre-timeout launched requests
        with self._state_lock:
            if (
                self._cutoff_active
                and self._cutoff_last_launched_request_id is not None
            ):
                if (
                    req_id is not None
                    and req_id <= self._cutoff_last_launched_request_id
                ):
                    self._num_pre_timeout_terminal += 1
                    logger.debug(
                        "Tail progress: pre_timeout_terminal=%d/%d",
                        self._num_pre_timeout_terminal,
                        self._cutoff_launched_requests,
                    )

    def get_duration_summary(self) -> Dict[str, float]:
        return {
            "Duration": self.duration,
            "Completed Requests Per Min": self.completed_requests_per_min,
        }

    def get_aggregated_summary(self) -> Dict[str, float]:
        return {
            **self.metric_store.get_aggregated_summary(),
            **self.get_duration_summary(),
        }

    def get_summary(self) -> Dict[str, float]:
        return {
            **self.metric_store.get_summary(),
            **self.get_duration_summary(),
        }

    def __str__(self) -> str:
        return "\n".join(
            [f"{k}: {v:.5f}" for k, v in self.get_aggregated_summary().items()]
            + [str(summary) for summary in self.metric_store.summaries.values()]
        )

    def __repr__(self) -> str:
        return self.__str__()

    def store_output(self):
        self.metric_store.store_output(self.output_dir)

    def request_stop(self) -> None:
        """Signal main loop to stop as soon as possible."""
        self._stop_event.set()

    def notify_error(self, exc: BaseException) -> None:
        """Record an error and request stop; main loop can re-raise."""
        with self._error_lock:
            self._error = exc
        self._error_event.set()
        self._stop_event.set()

    @property
    def error(self) -> Optional[BaseException]:
        with self._error_lock:
            return self._error

    @property
    def stop_requested(self) -> bool:
        """Return True if a stop has been requested (thread-safe)."""
        return self._stop_event.is_set()
