import logging
import threading
import time
from queue import Empty, Queue
from threading import Thread
from typing import List, Optional

from tqdm import tqdm  # type: ignore

from veeksha.core.dispatch_scheduler import DispatchScheduler
from veeksha.core.requests_launcher import RequestsLauncher
from veeksha.core.response import Response
from veeksha.dashboard.events import RequestCompletedEvent, RequestStartedEvent
from veeksha.dashboard.handler import emit_dashboard_event
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics

logger = init_logger(__name__)


class RequestDispatcher:
    """Handles dispatching of requests to worker queues."""

    def __init__(
        self,
        input_queue: Queue,
        service_metrics: ServiceMetrics,
        benchmark_id: str,
        telemetry_enabled: bool,
    ):
        self.input_queue = input_queue
        self.service_metrics = service_metrics
        self.benchmark_id = benchmark_id
        self.telemetry_enabled = telemetry_enabled

    def dispatch_request(self, request_config) -> None:
        """Dispatch a single request to workers."""
        self.service_metrics.register_launched_request()
        request_config.benchmark_id = self.benchmark_id
        self.input_queue.put(request_config)

        if self.telemetry_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Dispatched request {request_config.id}")

        # Emit dashboard event
        if request_config.id is not None:
            emit_dashboard_event(
                RequestStartedEvent(
                    request_id=request_config.id,
                    timestamp=time.time(),
                    input_tokens=request_config.prompt[1],
                    benchmark_id=self.benchmark_id,
                )
            )


def prefetch_requests(
    ready_queue: Queue,
    service_metrics: ServiceMetrics,
    request_generator: BaseRequestGenerator,
    generator_lock: threading.Lock,
    stop_event: threading.Event,
    thread_id: int,
) -> None:
    """Prefetch thread: Generate requests and push them to the ready queue.

    Multiple threads can run this function in parallel. The generator_lock ensures
    thread-safe access to the request generator.
    """
    logger.info(f"Prefetch thread {thread_id} starting")

    while not stop_event.is_set():
        # Generate next request (thread-safe)
        with generator_lock:
            # Check if we've generated enough requests
            if service_metrics.num_generated_requests >= service_metrics.max_requests:
                logger.info(f"Prefetch thread {thread_id}: max requests reached")
                break

            try:
                request_config = request_generator.get_request()
            except StopIteration:
                logger.info(f"Prefetch thread {thread_id}: generator exhausted")
                break

            # Handle special sentinel values
            if request_config.dispatch_delay == -1:
                logger.info(f"Prefetch thread {thread_id}: stop policy triggered")
                service_metrics.request_stop()
                stop_event.set()
                break
            elif request_config.dispatch_delay < 0:
                raise ValueError(
                    f"Invalid dispatch_delay '{request_config.dispatch_delay}' from generator"
                )

            service_metrics.num_generated_requests += 1
            requests_generated = service_metrics.num_generated_requests

        # Push to ready queue (outside lock to avoid holding it too long)
        ready_queue.put(request_config)

        if requests_generated % 1000 == 0:
            logger.debug(f"Prefetch progress: {requests_generated} requests generated")

    logger.info(f"Prefetch thread {thread_id} exiting")


def dispatch_requests(
    input_queue: Queue,
    ready_queue: Queue,
    service_metrics: ServiceMetrics,
    stop_event: threading.Event,
    scheduler: DispatchScheduler,
    req_launcher: RequestsLauncher,
    thread_id: int,
    benchmark_id: str = "default",
    telemetry_enabled: bool = False,
) -> None:
    """Dispatcher thread: Take requests from ready queue and dispatch them when ready.

    Multiple threads can run this function in parallel. The scheduler is thread-safe.
    """
    logger.info(f"Dispatcher thread {thread_id} starting")

    # Create dispatcher instance to handle request dispatching
    dispatcher = RequestDispatcher(
        input_queue=input_queue,
        service_metrics=service_metrics,
        benchmark_id=benchmark_id,
        telemetry_enabled=telemetry_enabled,
    )

    while not stop_event.is_set():
        # Check if there's a request ready to dispatch from scheduler
        ready = scheduler.pop_ready()
        if ready is not None:
            dispatcher.dispatch_request(ready)
            continue

        # Try to get a new request from the ready queue
        try:
            request_config = ready_queue.get(timeout=0.1)
        except Empty:
            # No new requests, check if we should exit
            if stop_event.is_set():
                break
            continue

        # Add to scheduler (handles dispatch timing and session sequencing)
        scheduler.add_request(request_config)

        # Try to dispatch immediately if it's ready
        ready = scheduler.pop_ready()
        if ready is not None:
            dispatcher.dispatch_request(ready)

    # Drain any remaining ready requests from scheduler (only one thread should do this)
    # Since multiple dispatcher threads may reach this point, they'll all try to drain
    # but that's okay - pop_ready is thread-safe and returns None when empty
    logger.info(f"Dispatcher thread {thread_id}: draining scheduler before exit")
    while True:
        ready = scheduler.pop_ready()
        if ready is None:
            break
        dispatcher.dispatch_request(ready)

    logger.info(f"Dispatcher thread {thread_id} exiting")


def process_results(
    output_queue: Queue,
    service_metrics: ServiceMetrics,
    generated_responses: List[Response],
    responses_lock: threading.Lock,
    pbar: tqdm,
    pbar_lock: threading.Lock,
    stop_event: threading.Event,
    scheduler: DispatchScheduler,
    thread_id: int,
) -> None:
    """Results processor thread: Process completed requests from workers.

    Multiple threads can run this function in parallel. Locks protect shared resources:
    - responses_lock: protects generated_responses list
    - pbar_lock: protects progress bar updates
    - service_metrics is already thread-safe
    - scheduler is already thread-safe
    """
    logger.info(f"Results processor thread {thread_id} starting")

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
                        f"Result processor drained for ~{DRAIN_MAX_EMPTY_POLLS * POLL_TIMEOUT_S:.1f}s after stop; exiting"
                    )
                    break
            continue

        if result is None:  # Sentinel value
            break

        request_metrics, generated_response = result
        service_metrics.add_request_metrics(request_metrics)

        # Notify scheduler about completion (for session-aware sequencing)
        success = (
            getattr(request_metrics, "error_code", None) is None
            and getattr(request_metrics, "error_msg", None) is None
        )
        scheduler.notify_completion(
            request_id=request_metrics.request_id,
            completed_at_monotonic=time.monotonic(),
            success=success,
        )

        if generated_response is not None:
            with responses_lock:
                generated_responses.append(generated_response)

        # Emit dashboard completion event
        if request_metrics.request_id is not None:
            emit_dashboard_event(
                RequestCompletedEvent(
                    request_id=str(request_metrics.request_id),
                    timestamp=time.time(),
                    final_metrics=request_metrics,
                    benchmark_id=request_metrics.benchmark_id,
                )
            )

        with pbar_lock:
            pbar.update(service_metrics.num_completed_requests - pbar.n)

    logger.info(f"Results processor thread {thread_id} exiting")
