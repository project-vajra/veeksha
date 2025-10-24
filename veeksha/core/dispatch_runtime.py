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
from revati.client.helper import create_thread_local_revati_client, get_time, sleep, is_revati_enabled

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
    prefetch_stats_lock = threading.Lock()
    prefetch_tick_counter = 0
    scheduled_since_log = 0
    total_scheduled = 0  # monotonic count of total requests added to scheduler
    create_thread_local_revati_client(f"veeksha-dispatcher-{str(uuid.uuid4())[:8]}", ClientType.ACTOR)
    prefetch_rate_window_start = get_time()
    next_prefetch_rate_log_time = get_time() + PREFETCH_RATE_LOG_INTERVAL_S

    def _can_send_request() -> bool:
        with prefetch_stats_lock:
            num_err_handled_snapshot = num_errored_requests_handled
        return should_send_new_request(service_metrics, num_err_handled_snapshot)

    def _prefetch_time_gate(now_pf: float) -> bool:
        nonlocal next_prefetch_time
        time_until_pf = scheduler.time_until_next_ready()
        prefetch_safe_threshold = max(PREFETCH_INTERVAL_S, NEAR_DEADLINE_WINDOW_S)
        safe_to_prefetch = (
            time_until_pf is None or time_until_pf >= prefetch_safe_threshold
        )
        if not safe_to_prefetch:
            sleep(0.001)
            return True
        if now_pf < next_prefetch_time:
            remaining = next_prefetch_time - now_pf
            if is_revati_enabled():
                sleep(remaining)
            else:
                if remaining > 0.002:
                    sleep(min(remaining - 0.0005, 0.002))
                else:
                    deadline = next_prefetch_time
                    while True:
                        now_spin = get_time()
                        if now_spin >= deadline or stop_event.is_set():
                            break
                        time.sleep(0)
            return True
        return False

    def _is_over_scheduled_cap(now_pf: float) -> bool:
        nonlocal next_prefetch_time
        with prefetch_stats_lock:
            unhandled_error_allowance = max(
                0,
                service_metrics.num_errored_requests - num_errored_requests_handled,
            )
            scheduled_cap = (
                service_metrics.max_requests
                + PREFETCH_SCHEDULE_SLACK
                + unhandled_error_allowance
            )
            current_total = total_scheduled
        if current_total >= scheduled_cap:
            next_prefetch_time = now_pf + PREFETCH_INTERVAL_S
            sleep(0.001)
            return True
        return False

    def _mark_prefetch_tick() -> None:
        nonlocal prefetch_tick_counter
        with prefetch_stats_lock:
            prefetch_tick_counter += 1

    def _try_prefetch_request() -> str:
        nonlocal generator_exhausted, scheduled_backlog, scheduled_since_log, total_scheduled
        blocked_pending_pf = scheduler.get_blocked_pending_count()
        with prefetch_stats_lock:
            effective_backlog_pf = max(0, scheduled_backlog - blocked_pending_pf)
            unhandled_error_allowance = max(
                0,
                service_metrics.num_errored_requests - num_errored_requests_handled,
            )
            scheduled_cap = (
                service_metrics.max_requests
                + PREFETCH_SCHEDULE_SLACK
                + unhandled_error_allowance
            )
            if total_scheduled >= scheduled_cap:
                return "break"
        if effective_backlog_pf >= MAX_PREFETCH_BACKLOG:
            return "break"

        try:
            request_config = request_generator.get_request()
        except StopIteration:
            generator_exhausted = True
            return "break"

        if request_config.dispatch_delay == -1:
            logger.info(
                "Benchmark ending early due to stop policy (generator sentinel received)."
            )
            service_metrics.request_stop()
            stop_event.set()
            return "break"
        elif request_config.dispatch_delay < 0:
            raise ValueError(
                f"Invalid request dispatch delay '{request_config.dispatch_delay}' from request metadata."
            )

        scheduler.add_request(request_config)
        with prefetch_stats_lock:
            scheduled_backlog += 1
            if telemetry_enabled:
                scheduled_since_log += 1
            total_scheduled += 1
        return "scheduled"

    def prefetch_loop() -> None:
        nonlocal next_prefetch_time, generator_exhausted
        create_thread_local_revati_client(f"veeksha-dispatcher-prefetch-loop-{str(uuid.uuid4())[:8]}", ClientType.ACTOR)

        while not stop_event.is_set():
            if generator_exhausted:
                break

            if not _can_send_request():
                break

            now_pf = get_time()

            if _prefetch_time_gate(now_pf):
                continue

            if _is_over_scheduled_cap(now_pf):
                continue

            if telemetry_enabled:
                _mark_prefetch_tick()

            status = _try_prefetch_request()
            if status == "break":
                break

            next_prefetch_time = now_pf + PREFETCH_INTERVAL_S

        # exit prefetch loop

    def _dispatch_ready_request(ready) -> None:
        nonlocal scheduled_backlog, num_errored_requests_handled
        service_metrics.register_launched_request()
        if service_metrics.num_requests > service_metrics.max_requests:
            with prefetch_stats_lock:
                num_errored_requests_handled += 1

        ready.benchmark_id = benchmark_id  # dashboard
        input_queue.put(ready)
        with prefetch_stats_lock:
            if scheduled_backlog > 0:
                scheduled_backlog -= 1
        if telemetry_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Dispatched request %s", ready.id)

        # Request ID should always be set by the generator
        assert ready.id is not None, f"Request {ready} has no ID"
        emit_dashboard_event(
            RequestStartedEvent(
                request_id=ready.id,
                timestamp=time.time(),
                input_tokens=ready.prompt[1],
                benchmark_id=benchmark_id,
            )
        )

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
        with prefetch_stats_lock:
            scheduled_backlog_snapshot = scheduled_backlog
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
        with prefetch_stats_lock:
            elapsed = max(1e-9, now - prefetch_rate_window_start)
            ticks_hz = prefetch_tick_counter / elapsed
            scheduled_rps = scheduled_since_log / elapsed
            logger.info(
                "Prefetch rate | ticks=%.1f Hz scheduled=%.1f req/s",
                ticks_hz,
                scheduled_rps,
            )
            prefetch_tick_counter = 0
            scheduled_since_log = 0
            prefetch_rate_window_start = now
            next_prefetch_rate_log_time = now + PREFETCH_RATE_LOG_INTERVAL_S

    def _spin_near_deadline(time_until: Optional[float]) -> bool:
        if time_until is None or time_until > NEAR_DEADLINE_WINDOW_S:
            return False
        deadline = get_time() + time_until
        
        if is_revati_enabled():
            sleep(time_until)
        else:
            while get_time() < deadline:
                ready_local = scheduler.pop_ready()
                if ready_local is not None:
                    _dispatch_ready_request(ready_local)
                    return True
                remaining = deadline - get_time()
                if remaining <= 0:
                    break
                sleep(min(remaining, 0.001))

        ready_local = scheduler.pop_ready()
        if ready_local is not None:
            _dispatch_ready_request(ready_local)
            return True
        return False

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
            next_spawn_time = get_time() + SPAWN_COOLDOWN_S

    def _maybe_warn_backlog(now: float, effective_backlog: int) -> None:
        nonlocal next_backlog_warn_time
        if effective_backlog < MAX_PREFETCH_BACKLOG or now < next_backlog_warn_time:
            return
        with prefetch_stats_lock:
            scheduled_backlog_snapshot2 = scheduled_backlog
        logger.info(
            "Effective prefetch backlog reached cap (%d). scheduled=%d blocked_pending=%d ready=%d ready_now=%d",
            MAX_PREFETCH_BACKLOG,
            scheduled_backlog_snapshot2,
            scheduler.get_blocked_pending_count(),
            scheduler.get_ready_count(),
            scheduler.get_ready_now_count(),
        )
        next_backlog_warn_time = now + BACKLOG_WARN_INTERVAL_S

    # Start prefetcher thread
    prefetch_thread = Thread(
        target=prefetch_loop, name="dispatch-prefetcher", daemon=True
    )
    prefetch_thread.start()

    while not stop_event.is_set():
        now = get_time()
        effective_backlog = 0  # only used with telemetry enabled
        if telemetry_enabled:
            _maybe_log_backlog(now)
            _maybe_log_prefetch_rate(now)

        # immediate dispatch
        ready = scheduler.pop_ready()
        if ready is not None:
            _dispatch_ready_request(ready)
            continue

        time_until = scheduler.time_until_next_ready()
        if _spin_near_deadline(time_until):
            continue

        # effective backlog ignores blocked session-followup requests (telemetry only)
        if telemetry_enabled:
            blocked_pending = scheduler.get_blocked_pending_count()
            with prefetch_stats_lock:
                effective_backlog = max(0, scheduled_backlog - blocked_pending)

        _maybe_auto_spawn_clients(now)

        if telemetry_enabled:
            _maybe_warn_backlog(get_time(), effective_backlog)

        # dispatch again after prefetch
        ready = scheduler.pop_ready()
        if ready is not None:
            _dispatch_ready_request(ready)
            continue

        # back off briefly
        time_until = scheduler.time_until_next_ready()
        sleep_time = 0.01 if time_until is None else min(max(time_until, 0.001), 0.1)
        sleep(sleep_time)

    # Join prefetcher on exit
    prefetch_thread.join(timeout=1.0)


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
        # notify scheduler about completion for session-aware sequencing
        success = (
            getattr(request_metrics, "error_code", None) is None
            and getattr(request_metrics, "error_msg", None) is None
        )
        scheduler.notify_completion(
            request_id=request_metrics.request_id,
            completed_at_monotonic=get_time(),
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
