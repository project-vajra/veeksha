import logging
import threading
import time
from multiprocessing import Queue
from queue import Empty
from threading import Thread
import uuid
from typing import List, Optional

from tqdm import tqdm  # type: ignore

from revati.client import ClientType  # type: ignore
from revati.client.helper import create_thread_local_revati_client, get_virtual_time, advance_time, is_revati_enabled, wait_for_clock_update, yield_barrier, rejoin_barrier


from veeksha.core.dispatch_scheduler import DispatchScheduler
from veeksha.core.requests_launcher import RequestsLauncher
from veeksha.core.response import Response
from veeksha.dashboard.events import RequestCompletedEvent, RequestStartedEvent
from veeksha.dashboard.handler import emit_dashboard_event
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics

logger = init_logger(__name__)


PREFETCH_INTERVAL_S = 0.004  # 250 rps
MAX_PREFETCH_BACKLOG = 10000
NEAR_DEADLINE_WINDOW_S = 0.010
BACKLOG_LOG_INTERVAL_S = 1.0
BACKLOG_WARN_INTERVAL_S = 5.0
SPAWN_SUPPRESSION_INTERVAL_S = 10.0
SPAWN_COOLDOWN_S = 1.0
PREFETCH_RATE_LOG_INTERVAL_S = 2.0
PREFETCH_SCHEDULE_SLACK = 512  # number of requests to prefetch beyond max_requests


def should_send_new_request(
    service_metrics: ServiceMetrics, num_errored_requests_handled: int
) -> bool:
    """Check if a request should be sent based on the current state of the service."""
    return (service_metrics.num_requests < service_metrics.max_requests) or (
        service_metrics.num_requests >= service_metrics.max_requests
        and num_errored_requests_handled < service_metrics.num_errored_requests
    )


def dispatch_requests(
    input_queue: Queue,
    service_metrics: ServiceMetrics,
    request_generator: BaseRequestGenerator,
    stop_event: threading.Event,
    scheduler: DispatchScheduler,
    req_launcher: RequestsLauncher,
    benchmark_id: str = "default",
    telemetry_enabled: bool = False,
) -> None:
    """Thread function to generate and dispatch requests."""
    num_errored_requests_handled = 0

    # scheduler provided by caller
    next_prefetch_time = 0.0
    generator_exhausted = False
    scheduled_backlog = 0
    next_backlog_log_time = 0.0
    next_backlog_warn_time = 0.0
    next_spawn_time = 0.0
    next_spawn_suppression_time = 0.0
    prefetch_tick_counter = 0
    scheduled_since_log = 0
    total_scheduled = 0  # monotonic count of total requests added to scheduler

    if is_revati_enabled():
        create_thread_local_revati_client(f"veeksha-dispatcher-{str(uuid.uuid4())[:8]}", ClientType.ACTOR)

    prefetch_rate_window_start = get_virtual_time()
    next_prefetch_rate_log_time = get_virtual_time() + PREFETCH_RATE_LOG_INTERVAL_S

    def _can_send_request() -> bool:
        return should_send_new_request(service_metrics, num_err_handled_snapshot)

    def _mark_prefetch_tick() -> None:
        nonlocal prefetch_tick_counter

    def _try_prefetch_request() -> str:
        nonlocal generator_exhausted, scheduled_backlog, scheduled_since_log, total_scheduled
        blocked_pending_pf = scheduler.get_blocked_pending_count()
        if total_scheduled >= service_metrics.max_requests:
            generator_exhausted = True
            return "break"

        try:
            request_config = request_generator.get_request()
        except StopIteration:
            generator_exhausted = True
            return "break"

        scheduler.add_request(request_config)
        total_scheduled += 1
        return "scheduled"

    def _dispatch_ready_request(ready) -> None:
        nonlocal scheduled_backlog, num_errored_requests_handled
        service_metrics.register_launched_request()

        ready.benchmark_id = benchmark_id  # dashboard
        input_queue.put(ready)


    def _maybe_log_backlog(now: float) -> None:
        nonlocal next_backlog_log_time
        if now < next_backlog_log_time:
            return
        blocked_pending_snapshot = scheduler.get_blocked_pending_count()
        ready_count_snapshot = scheduler.get_ready_count()
        ready_now_snapshot = scheduler.get_ready_now_count()
        try:
            input_queue_size = input_queue.qsize()
        except NotImplementedError:
            input_queue_size = -1

        effective_backlog_snapshot = max(
            0, scheduled_backlog_snapshot - blocked_pending_snapshot
        )
        logger.info(
            "Prefetch backlog | scheduled=%d effective=%d blocked_pending=%d ready=%d ready_now=%d in_q=%d",
            scheduled_backlog_snapshot,
            effective_backlog_snapshot,
            blocked_pending_snapshot,
            ready_count_snapshot,
            ready_now_snapshot,
            input_queue_size,
        )
        next_backlog_log_time = now + BACKLOG_LOG_INTERVAL_S

    def _maybe_log_prefetch_rate(now: float) -> None:
        nonlocal prefetch_rate_window_start, next_prefetch_rate_log_time
        nonlocal prefetch_tick_counter, scheduled_since_log
        if now < next_prefetch_rate_log_time:
            return

    def _maybe_auto_spawn_clients(now: float) -> None:
        nonlocal next_spawn_time, next_spawn_suppression_time
        if not req_launcher.client_config.auto_spawn_new_clients:
            return
        inflight = service_metrics.num_requests - service_metrics.num_completed_requests
        total_slots = req_launcher.get_total_slots()
        try:
            input_queue_size_now = input_queue.qsize()
        except NotImplementedError:
            input_queue_size_now = -1  # sentinel for unknown size
        available_slots = max(0, total_slots - inflight)
        has_queued_or_unknown = (input_queue_size_now > 0) or (
            input_queue_size_now == -1
        )
        if has_queued_or_unknown and available_slots == 0 and now >= next_spawn_time:
            if req_launcher.can_spawn_more():
                logger.info(
                    "Auto-spawning new client: reqs_queued=%d inflight=%d total_slots=%d",
                    input_queue_size_now,
                    inflight,
                    total_slots,
                )
                req_launcher.spawn_new_client()
            else:
                if now >= next_spawn_suppression_time:
                    logger.info(
                        "Client spawn suppressed: at max_clients=%s (reqs_queued=%d inflight=%d total_slots=%d)",
                        str(req_launcher.client_config.max_clients),
                        input_queue_size_now,
                        inflight,
                        total_slots,
                    )
                    next_spawn_suppression_time = now + SPAWN_SUPPRESSION_INTERVAL_S
            next_spawn_time = get_virtual_time() + SPAWN_COOLDOWN_S

    def _maybe_warn_backlog(now: float, effective_backlog: int) -> None:
        nonlocal next_backlog_warn_time
        if effective_backlog < MAX_PREFETCH_BACKLOG or now < next_backlog_warn_time:
            return

        logger.info(
            "Effective prefetch backlog reached cap (%d). scheduled=%d blocked_pending=%d ready=%d ready_now=%d",
            MAX_PREFETCH_BACKLOG,
            scheduled_backlog_snapshot2,
            scheduler.get_blocked_pending_count(),
            scheduler.get_ready_count(),
            scheduler.get_ready_now_count(),
        )
        next_backlog_warn_time = now + BACKLOG_WARN_INTERVAL_S
    
    while not generator_exhausted:
        _try_prefetch_request()

    print("generator_exhausted, loaded {} requests".format(total_scheduled), flush=True)

    while not stop_event.is_set():
        now = get_virtual_time()

        # immediate dispatch
        ready = scheduler.pop_ready()
        if ready is not None:
            _dispatch_ready_request(ready)
            yield_barrier()
            wait_for_clock_update(40) # in ms
            rejoin_barrier()
            continue

        time_until = scheduler.time_until_next_ready()
        if time_until is None:
            break
        advance_time(time_until)
        _maybe_auto_spawn_clients(now)


def process_results(
    output_queue: Queue,
    service_metrics: ServiceMetrics,
    generated_responses: List[Response],
    pbar: tqdm,
    stop_event: threading.Event,
    scheduler: DispatchScheduler,
) -> None:
    """Thread function to process results from the output queue."""
    # On stop, attempt to drain for a short grace period, then exit
    POLL_TIMEOUT_S = 0.1
    DRAIN_MAX_EMPTY_POLLS = 50  # ~5s
    consecutive_empty_polls_after_stop = 0
    while not stop_event.is_set() or (
        service_metrics.error is None
        and service_metrics.num_completed_requests < service_metrics.num_requests
    ):
        try:
            result = output_queue.get(timeout=POLL_TIMEOUT_S)
            consecutive_empty_polls_after_stop = 0
        except Empty:
            if stop_event.is_set():
                consecutive_empty_polls_after_stop += 1
                if consecutive_empty_polls_after_stop >= DRAIN_MAX_EMPTY_POLLS:
                    logger.info(
                        "Result processor drained for ~%.1fs after stop; exiting.",
                        DRAIN_MAX_EMPTY_POLLS * POLL_TIMEOUT_S,
                    )
                    break
            continue

        if result is None:  # Sentinel check
            break

        request_metrics, generated_response = result
        service_metrics.add_request_metrics(request_metrics)

        # Fail fast on errors to avoid stuck experiments
        if request_metrics.error_code or request_metrics.error_msg:
            error = RuntimeError(
                f"Request failed: {request_metrics.error_msg} (code: {request_metrics.error_code})"
            )
            service_metrics.notify_error(error)

        # notify scheduler about completion for session-aware sequencing
        success = (
            getattr(request_metrics, "error_code", None) is None
            and getattr(request_metrics, "error_msg", None) is None
        )
        scheduler.notify_completion(
            request_id=request_metrics.request_id,
            completed_at_monotonic=get_virtual_time(),
            success=success,
        )
        if generated_response is not None:
            generated_responses.append(generated_response)

        # Emit completion event - ensure request_id is set
        assert (
            request_metrics.request_id is not None
        ), f"Request metrics has no ID: {request_metrics}"
        emit_dashboard_event(
            RequestCompletedEvent(
                request_id=str(request_metrics.request_id),
                timestamp=time.time(),
                final_metrics=request_metrics,
                benchmark_id=request_metrics.benchmark_id,
            )
        )

        # TODO: maybe add benchmark status event here?

        pbar.update(service_metrics.num_completed_requests - pbar.n)
