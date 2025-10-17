import multiprocessing
import os
import platform
import random
import threading
import time
from multiprocessing import Queue
from queue import Empty
from threading import Thread
from typing import List, Optional, TypedDict

from tqdm import tqdm  # type: ignore

try:
    from revati.core.time_sync import Client as RevatiClient
except ImportError:
    RevatiClient = None

from veeksha.benchmark_data_utils import (
    load_corpus,
    store_generated_texts,
    store_lmeval_results,
)
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import ClientConfig
from veeksha.config.utils import prepare_benchmark_output_dir
from veeksha.core.dispatch_scheduler import DispatchScheduler
from veeksha.core.hf_utils import get_tokenizer
from veeksha.core.requests_launcher import RequestsLauncher
from veeksha.core.response import Response
from veeksha.core.seeding import (
    SeedManager,
)
from veeksha.dashboard.events import (
    BenchmarkStatusEvent,
    RequestCompletedEvent,
    RequestStartedEvent,
)
from veeksha.dashboard.handler import (
    emit_dashboard_event,
    get_dashboard_event_processor,
    init_dashboard_event_processor,
    stop_dashboard_event_processor,
)
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.generators.request_generator.generator_registry import (
    RequestGeneratorRegistry,
)
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics
from veeksha.types import RequestGeneratorType

logger = init_logger(__name__)

PREFETCH_BATCH_SIZE = 1
PREFETCH_INTERVAL_S = 0.001
MAX_PREFETCH_BACKLOG = 20
NEAR_DEADLINE_WINDOW_S = 0.010


def setup_api_environment(
    api_key=None,
    api_url=None,
):
    """Set up environment variables for OpenAI API"""
    assert api_key is not None, "API key is required"
    assert api_url is not None, "API URL is required"
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_url


def _probe_min_tokens_param_support(client_config: ClientConfig) -> bool:
    """Probe if server accepts the configured min token parameter."""
    import os

    import requests  # type: ignore

    min_param: Optional[str] = client_config.min_tokens_param
    if not min_param:
        return False

    base_url = os.environ.get("OPENAI_API_BASE")
    if not base_url:
        logger.warning("OPENAI_API_BASE not set; cannot probe min token parameter.")
        return False
    if not base_url.endswith("/"):
        base_url = base_url + "/"

    url = base_url + (client_config.address_append_value or "chat/completions")
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
        "Content-Type": "application/json",
    }

    body = {
        "model": client_config.model,
        "stream": False,
        "max_tokens": 1,
    }
    if client_config.llm_api == "openai_completions":
        body["prompt"] = "Hello"
    else:
        body["messages"] = [{"role": "user", "content": "Hello"}]

    def send_probe_request(param_value):
        test_body = body.copy()
        test_body[min_param] = param_value
        try:
            resp = requests.post(url, headers=headers, json=test_body, timeout=10)
            return 200 <= resp.status_code < 300
        except Exception:
            return False

    if not send_probe_request(1):
        logger.warning(
            f"Server rejected parameter '{min_param}'; falling back to prompt control."
        )
        return False

    if not send_probe_request({"invalid": "type"}):
        return True

    return False


def _initialize_min_tokens_support(benchmark_config: BenchmarkConfig) -> None:
    """Initialize min tokens parameter support by probing the server.

    This function probes the server to determine if it supports the configured
    min_tokens_param. If not supported, it disables the parameter and logs
    a warning about falling back to prompt-based control.
    """
    if benchmark_config.client_config.min_tokens_param:
        is_supported = _probe_min_tokens_param_support(benchmark_config.client_config)
        min_tokens_param = benchmark_config.client_config.min_tokens_param
        if not is_supported:
            object.__setattr__(benchmark_config.client_config, "min_tokens_param", None)
            logger.warning(
                f"min_tokens_param '{min_tokens_param}' not supported by server; switching to prompt-based minimum token control. This will include, in each request, an instruction to generate at least the requested number of tokens. Might lead to inaccurate lengths being generated."
            )
        else:
            logger.info(
                f"min_tokens_param '{min_tokens_param}' supported in request body."
            )


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
    revati_client: Optional["RevatiClient"] = None,
    benchmark_id: str = "default",
) -> None:
    """Thread function to generate and dispatch requests.

    Args:
        input_queue: Queue for dispatching requests
        service_metrics: Metrics tracker
        request_generator: Generator for creating requests
        stop_event: Event to signal thread termination
        scheduler: Scheduler for managing request dispatch timing
        revati_client: Optional revati client for time synchronization
    """
    num_errored_requests_handled = 0

    # scheduler provided by caller
    next_prefetch_time = 0.0
    generator_exhausted = False
    scheduled_backlog = 0

    def _dispatch_ready_request(ready) -> None:
        nonlocal scheduled_backlog, num_errored_requests_handled
        service_metrics.register_launched_request()
        if service_metrics.num_requests > service_metrics.max_requests:
            num_errored_requests_handled += 1
        input_queue.put(ready)
        if scheduled_backlog > 0:
            scheduled_backlog -= 1
        logger.debug(f"Dispatched request {ready.id}")

        ready.benchmark_id = benchmark_id
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

    while not stop_event.is_set():
        now = time.monotonic()

        # immediate dispatch
        ready = scheduler.pop_ready()
        if ready is not None:
            _dispatch_ready_request(ready)
            continue

        time_until = scheduler.time_until_next_ready()

        # short spin near deadlines (up to 10ms)
        if time_until is not None and time_until <= NEAR_DEADLINE_WINDOW_S:
            deadline = time.monotonic() + time_until
            while time.monotonic() < deadline:
                ready = scheduler.pop_ready()
                if ready is not None:
                    _dispatch_ready_request(ready)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.001))
            if ready is not None:
                continue
            # immediate check at boundary
            ready = scheduler.pop_ready()
            if ready is not None:
                _dispatch_ready_request(ready)
                continue

        # prefetch away from near deadlines
        if (
            (not generator_exhausted)
            and should_send_new_request(service_metrics, num_errored_requests_handled)
            and (scheduled_backlog < MAX_PREFETCH_BACKLOG)
        ):
            # gate prefetch away from near deadlines
            now = time.monotonic()
            time_until = scheduler.time_until_next_ready()
            prefetch_safe_threshold = max(PREFETCH_INTERVAL_S, NEAR_DEADLINE_WINDOW_S)
            if (time_until is None or time_until >= prefetch_safe_threshold) and (
                now >= next_prefetch_time
            ):
                for _ in range(PREFETCH_BATCH_SIZE):
                    if scheduled_backlog >= MAX_PREFETCH_BACKLOG:
                        break
                    try:
                        request_config = request_generator.get_request()
                    except StopIteration:
                        # stop prefetching but keep dispatching already-scheduled
                        generator_exhausted = True
                        break

                    if request_config.dispatch_delay == -1:
                        logger.info(
                            "Benchmark ending early due to stop policy (generator sentinel received)."
                        )
                        service_metrics.request_stop()
                        stop_event.set()
                        break
                    elif request_config.dispatch_delay < 0:
                        raise ValueError(
                            f"Invalid request dispatch delay '{request_config.dispatch_delay}' from request metadata."
                        )

                    scheduler.add_request(request_config)
                    scheduled_backlog += 1
                next_prefetch_time = now + PREFETCH_INTERVAL_S
            else:
                # TODO: may be wrong place Can't prefetch now (near deadline or not time yet) - jump time until next request is ready
                if revati_client is not None and time_until is not None and time_until > 0:
                    revati_client.time_jump(time_until)

        # dispatch again after prefetch
        ready = scheduler.pop_ready()
        if ready is not None:
            if stop_event.is_set():
                logger.info("Stop requested, not dispatching request %d", ready.id)
                break

            _dispatch_ready_request(ready)
            continue

        # back off briefly
        time_until = scheduler.time_until_next_ready()
        sleep_time = 0.01 if time_until is None else min(max(time_until, 0.001), 0.1)

        # Use revati time synchronization if available, otherwise fall back to time.sleep
        if revati_client is not None:
            # Jump to next dispatch time if there are scheduled requests
            if time_until is not None and time_until > 0:
                revati_client.time_jump(time_until)
            else:
                # Wait for interrupts (request completions, etc.)
                revati_client.time_wait_until_interrupt()
        else:
            time.sleep(sleep_time)


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
            completed_at_monotonic=time.monotonic(),
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


def run_main_loop(
    benchmark_config: BenchmarkConfig,
    request_generator: BaseRequestGenerator,
    service_metrics: ServiceMetrics,
    generated_responses: List[Response],
    pbar: tqdm,
    benchmark_id: str = "default",
):
    """Run the main loop for the benchmark."""

    logger.info("Starting the main loop.")

    # Create queues for communication
    input_queue = Queue()
    output_queue = Queue()
    stop_event = threading.Event()
    scheduler = DispatchScheduler()

    # Initialize revati client if enabled
    revati_client = None
    enable_revati = getattr(benchmark_config, "enable_revati_client", False)
    revati_server_address = getattr(
        benchmark_config, "revati_server_address", "ipc:///tmp/revati_time_sync_default.sock"
    )

    if enable_revati and RevatiClient:
        try:
            logger.info(
                f"Initializing Revati client at {revati_server_address} for dispatch timing"
            )
            revati_client = RevatiClient(
                server_address=revati_server_address, client_name="Dispatch"
            )
            revati_client.connect()
            revati_client.register()
            revati_client.start_simulation()
            logger.info("Revati dispatch client initialized and simulation started")
        except Exception as e:
            logger.warning(
                f"Failed to initialize Revati client: {e}. Falling back to normal timing."
            )
            revati_client = None
    elif enable_revati and not RevatiClient:
        logger.warning(
            "Revati client requested but not available. Install revati to enable time synchronization."
        )

    # Initialize request launcher
    req_launcher = RequestsLauncher(
        client_config=benchmark_config.client_config,
        input_queue=input_queue,
        output_queue=output_queue,
    )

    # Start the request launcher processes
    req_launcher.start()

    # Create and start producer-consumer threads
    dispatcher_thread = Thread(
        target=dispatch_requests,
        args=(
            input_queue,
            service_metrics,
            request_generator,
            stop_event,
            scheduler,
            revati_client,
            benchmark_id,
        ),
    )

    processor_thread = Thread(
        target=process_results,
        args=(
            output_queue,
            service_metrics,
            generated_responses,
            pbar,
            stop_event,
            scheduler,
        ),
    )

    dispatcher_thread.start()
    processor_thread.start()

    # Monitor and wait for completion
    with service_metrics:
        while not service_metrics.should_stop():
            time.sleep(0.1)
        logger.info("Stopping the main loop.")
        if service_metrics.stop_requested and service_metrics.error is None:
            logger.info(
                "Main loop exited due to stop policy; partial metrics will be saved."
            )

    # Signal threads to stop and wait for completion
    stop_event.set()
    dispatcher_thread.join()

    # Wait for all client processes to terminate
    req_launcher.wait_for_clients()

    # Signal the results processor to finish after draining and join it
    output_queue.put(None)
    processor_thread.join()

    pbar.close()

    # Clean up revati client if it was initialized
    if revati_client is not None:
        try:
            logger.info("Finalizing and disconnecting Revati dispatch client")
            revati_client.finalize()
            revati_client.disconnect()
            logger.info("Revati dispatch client cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error cleaning up Revati client: {e}")

    if service_metrics.error is None:
        logger.info("Main loop completed.")
    else:
        raise service_metrics.error


def run_benchmark(
    benchmark_config: BenchmarkConfig,
):
    """Run the benchmark and return the in-memory metrics object.

    Args:
        benchmark_config: The benchmark configuration.

    Returns:
        ServiceMetrics containing the collected metrics (including the `MetricStore`).
    """

    prepare_benchmark_output_dir(benchmark_config)

    # Ensure reproducibility across all execution paths
    random.seed(benchmark_config.seed)

    # Generate unique benchmark ID from output directory
    benchmark_id = os.path.basename(benchmark_config.metrics_config.output_dir)
    logger.info(
        f"Benchmark ID: {benchmark_id}, Output directory: {benchmark_config.metrics_config.output_dir}"
    )

    setup_api_environment(
        api_key=benchmark_config.api_key,
        api_url=benchmark_config.api_url,
    )

    _initialize_min_tokens_support(benchmark_config)

    # Dashboard initialization is now handled by the caller
    # (either run_benchmark_with_dashboard or run_benchmark_console_only)

    generated_responses: List[Response] = []

    assert (
        benchmark_config.client_config.tokenizer is not None
    ), "Tokenizer is required."

    tokenizer = get_tokenizer(
        tokenizer_name=benchmark_config.client_config.tokenizer,
        trust_remote_code=True,
    )

    request_generator_params = {}
    request_generator_config_type = benchmark_config.request_generator_config.get_type()

    if (
        request_generator_config_type == RequestGeneratorType.SYNTHETIC
        or request_generator_config_type == RequestGeneratorType.TRACE
    ):
        request_generator_params = {
            "corpus_lines": load_corpus(),
        }

    seed_manager = SeedManager(benchmark_config.seed)

    request_generator = RequestGeneratorRegistry.get(
        benchmark_config.request_generator_config.get_type(),
        config=benchmark_config.request_generator_config,
        tokenizer=tokenizer,
        client_config=benchmark_config.client_config,
        seed_manager=seed_manager,
        **request_generator_params,
    )

    max_requests = (
        request_generator.num_requests
        if benchmark_config.request_generator_config.get_type()
        == RequestGeneratorType.LMEVAL
        else benchmark_config.max_completed_requests
    )
    # Disable tqdm progress bar if dashboard is enabled to prevent output conflicts
    pbar = tqdm(total=max_requests, disable=benchmark_config.dashboard_config.enabled)

    service_metrics = ServiceMetrics(
        max_requests=max_requests,
        timeout=benchmark_config.timeout,
        metrics_config=benchmark_config.metrics_config,
    )

    run_main_loop(
        benchmark_config=benchmark_config,
        request_generator=request_generator,
        service_metrics=service_metrics,
        generated_responses=generated_responses,
        pbar=pbar,
        benchmark_id=benchmark_id,
    )

    service_metrics.store_output()
    logger.info(f"Metrics stored to {service_metrics.output_dir}")

    store_generated_texts(service_metrics.output_dir, generated_responses)

    # lm-eval specific
    if (
        benchmark_config.request_generator_config.get_type()
        == RequestGeneratorType.LMEVAL
    ):
        request_generator.get_responses(generated_responses)
        lmeval_results = request_generator.evaluate()

        store_lmeval_results(service_metrics.output_dir, lmeval_results)

    # Emit benchmark completion event for dashboard
    # This is done after all processing is complete to ensure accurate counts
    if service_metrics.start_time and service_metrics.end_time:
        elapsed = service_metrics.end_time - service_metrics.start_time
    else:
        elapsed = 0.0

    logger.debug(
        f"Emitting BenchmarkStatusEvent: total={service_metrics.num_requests}, "
        f"completed={service_metrics.num_completed_requests}, "
        f"errored={service_metrics.num_errored_requests}"
    )

    emit_dashboard_event(
        BenchmarkStatusEvent(
            benchmark_id=benchmark_id,
            total_requests=service_metrics.num_requests,
            completed_requests=service_metrics.num_completed_requests,
            errored_requests=service_metrics.num_errored_requests,
            active_requests=0,  # All requests done at this point
            current_qps=(
                service_metrics.num_completed_requests / elapsed if elapsed > 0 else 0.0
            ),
            elapsed_time=elapsed,
        )
    )

    return service_metrics


class BenchmarkResultContainer(TypedDict):
    service_metrics: Optional[ServiceMetrics]
    error: Optional[Exception]


def run_benchmark_with_dashboard(benchmark_config: BenchmarkConfig):
    """Run benchmark with TUI dashboard in main thread.

    The benchmark runs in a background thread while the TUI runs in the main thread.
    This is required because Textual needs to register signal handlers which can only
    be done in the main thread.
    """
    logger.info("Starting TUI dashboard with benchmark")

    # Ensure environment variables are set to suppress console output
    os.environ["VEEKSHA_SUPPRESS_CONSOLE_LOGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize dashboard event processor without frontend
    dashboard_state = init_dashboard_event_processor(
        enabled=benchmark_config.dashboard_config.enabled,
        enable_frontend=False,  # We'll launch TUI manually in main thread
        max_queue_size=benchmark_config.dashboard_config.max_queue_size,
        max_live_requests=benchmark_config.dashboard_config.max_live_requests,
        chart_window_seconds=benchmark_config.dashboard_config.chart_window_seconds,
    )

    # Result container to capture service_metrics from background thread
    result_container: BenchmarkResultContainer = {
        "service_metrics": None,
        "error": None,
    }

    def run_benchmark_thread():
        """Run benchmark in background thread"""
        try:
            service_metrics = run_benchmark(benchmark_config)
            result_container["service_metrics"] = service_metrics
        except Exception as e:
            result_container["error"] = e
            logger.error(f"Benchmark error: {e}", exc_info=True)

    # Start benchmark in background thread
    benchmark_thread = Thread(target=run_benchmark_thread, daemon=False)
    benchmark_thread.start()

    # Give benchmark a moment to initialize
    time.sleep(0.5)

    # Run TUI in main thread (blocking)
    if dashboard_state:
        from veeksha.dashboard.tui_dashboard import run_dashboard_tui

        run_dashboard_tui(dashboard_state)

    # Wait for benchmark thread to complete
    benchmark_thread.join()

    # Check for errors
    if result_container["error"]:
        raise result_container["error"]

    return result_container["service_metrics"]


def run_benchmark_console_only(
    benchmark_config: BenchmarkConfig, stop_processor_after: bool = True
):
    """Run benchmark with console-only output

    Args:
        benchmark_config: Configuration for the benchmark
        stop_processor_after: If True, stop the dashboard processor after this benchmark.
                            Set to False when running multiple benchmarks sequentially.
    """
    try:
        # Initialize dashboard if enabled (skip if already initialized)
        if (
            benchmark_config.dashboard_config.enabled
            and not get_dashboard_event_processor()
        ):
            init_dashboard_event_processor(
                enabled=True,
                enable_frontend=False,
                max_queue_size=benchmark_config.dashboard_config.max_queue_size,
                max_live_requests=benchmark_config.dashboard_config.max_live_requests,
                chart_window_seconds=benchmark_config.dashboard_config.chart_window_seconds,
            )

        service_metrics = run_benchmark(benchmark_config=benchmark_config)
        return service_metrics
    finally:
        if stop_processor_after:
            stop_dashboard_event_processor()


if __name__ == "__main__":
    # Use spawn mode when EventLoopIntegration is enabled to avoid ZMQ fork issues
    # Fork mode copies ZMQ socket file descriptors from parent, but ZMQ context is not fork-safe
    # This causes deadlocks in child processes when using revati time synchronization
    if os.getenv("REVATI_EVENT_LOOP_INTEGRATION") == "1":
        multiprocessing.set_start_method("spawn", force=True)
    elif platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    benchmark_configs = BenchmarkConfig.create_from_cli_args()

    # Check if dashboard is enabled
    has_dashboard_enabled = any(bc.dashboard_config.enabled for bc in benchmark_configs)
    if has_dashboard_enabled:
        # Set environment variable to suppress console logging in child processes
        os.environ["VEEKSHA_SUPPRESS_CONSOLE_LOGS"] = "1"
        # Suppress tokenizers parallelism warning when forking processes
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Remove stream handlers from loggers (but don't redirect stdout/stderr yet)
        # This allows the TUI to start properly
        import logging as log_module

        # Remove handlers from root logger
        root_logger = log_module.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, log_module.StreamHandler):
                root_logger.removeHandler(handler)

        # Remove handlers from veeksha logger specifically (which has its own handler)
        veeksha_logger = log_module.getLogger("veeksha")
        for handler in veeksha_logger.handlers[:]:
            if isinstance(handler, log_module.StreamHandler):
                veeksha_logger.removeHandler(handler)

    if len(benchmark_configs) > 1:
        logger.info(
            f"Running {len(benchmark_configs)} benchmark configurations sequentially."
        )
        logger.info("Using console dashboard for multiple configurations.")

    try:
        # single or multiple configs
        if has_dashboard_enabled and len(benchmark_configs) == 1:
            run_benchmark_with_dashboard(benchmark_configs[0])
        elif has_dashboard_enabled and len(benchmark_configs) > 1:
            # Multiple benchmarks with dashboard - run all in thread, then show TUI
            logger.info(f"Running {len(benchmark_configs)} benchmarks with dashboard")

            # Ensure environment variables are set
            os.environ["VEEKSHA_SUPPRESS_CONSOLE_LOGS"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # Initialize dashboard in main thread to avoid races
            first = benchmark_configs[0]
            dashboard_state = init_dashboard_event_processor(
                enabled=True,
                enable_frontend=False,
                max_queue_size=first.dashboard_config.max_queue_size,
                max_live_requests=first.dashboard_config.max_live_requests,
                chart_window_seconds=first.dashboard_config.chart_window_seconds,
            )

            def run_all_benchmarks():
                """Run all benchmarks sequentially in background"""
                for i, benchmark_config in enumerate(benchmark_configs):
                    logger.info(f"Starting benchmark {i+1}/{len(benchmark_configs)}")
                    is_last = i == len(benchmark_configs) - 1
                    run_benchmark_console_only(
                        benchmark_config, stop_processor_after=is_last
                    )
                    logger.info(f"Completed benchmark {i+1}/{len(benchmark_configs)}")

            # Start all benchmarks in background thread
            benchmark_thread = Thread(target=run_all_benchmarks, daemon=False)
            benchmark_thread.start()

            # Launch TUI dashboard
            if dashboard_state:
                from veeksha.dashboard.tui_dashboard import run_dashboard_tui

                run_dashboard_tui(dashboard_state)

            # Wait for all benchmarks to complete
            benchmark_thread.join()
            logger.info("All benchmarks completed")
        else:
            # No dashboard - console only mode
            for i, benchmark_config in enumerate(benchmark_configs):
                print(f"Running benchmark with config: {benchmark_config}")
                if len(benchmark_configs) > 1:
                    logger.info(f"Starting benchmark {i+1}/{len(benchmark_configs)}")

                is_last = i == len(benchmark_configs) - 1
                run_benchmark_console_only(
                    benchmark_config, stop_processor_after=is_last
                )

                if len(benchmark_configs) > 1:
                    logger.info(f"Completed benchmark {i+1}/{len(benchmark_configs)}")
    finally:
        logger.info("All benchmarks completed.")
        # Ensure processor is stopped even if there's an exception
        stop_dashboard_event_processor()
