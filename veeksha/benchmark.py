import multiprocessing
import os
import platform
import random
import threading
import time
from multiprocessing import Queue
from threading import Thread
from typing import List, Optional, TypedDict

from tqdm import tqdm  # type: ignore

from veeksha.benchmark_data_utils import (
    load_corpus,
    store_generated_texts,
    store_lmeval_results,
)
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import ClientConfig
from veeksha.config.utils import prepare_benchmark_output_dir
from veeksha.core.dispatch_runtime import dispatch_requests, process_results
from veeksha.core.dispatch_scheduler import DispatchScheduler
from veeksha.core.hf_utils import get_tokenizer
from veeksha.core.requests_launcher import RequestsLauncher
from veeksha.core.response import Response
from veeksha.core.seeding import (
    SeedManager,
)
from veeksha.dashboard.events import (
    BenchmarkStatusEvent,
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
MAX_PREFETCH_BACKLOG = 50000
NEAR_DEADLINE_WINDOW_S = 0.010
BACKLOG_LOG_INTERVAL_S = 1.0
BACKLOG_WARN_INTERVAL_S = 5.0
SPAWN_SUPPRESSION_INTERVAL_S = 10.0
SPAWN_COOLDOWN_S = 1.0
PREFETCH_RATE_LOG_INTERVAL_S = 2.0
PREFETCH_SCHEDULE_SLACK = 512  # number of requests to prefetch beyond max_requests


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
            req_launcher,
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
    # Expose output directory to child processes for iterative trace writes
    os.environ["VEEKSHA_OUTPUT_DIR"] = benchmark_config.metrics_config.output_dir
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    if platform.system() == "Darwin":
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
