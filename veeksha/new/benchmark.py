import os
import threading
import time
from queue import Queue
from typing import List, Optional

from tqdm import tqdm  # type: ignore

from veeksha.config.utils import prepare_benchmark_output_dir
from veeksha.core.context import BenchmarkContext
from veeksha.core.dispatch_scheduler import DispatchScheduler
from veeksha.core.response import Response
from veeksha.core.thread_pool import ThreadPoolManager
from veeksha.core.workers import DispatchWorker, PrefetchWorker, ResultsProcessorWorker
from veeksha.core.workers.request_runner_manager import RequestRunnerManager
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics
from veeksha.new.config.benchmark import BenchmarkConfig
from veeksha.new.config.client import ClientConfig
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.session import print_session
from veeksha.new.core.tokenizer import (
    TokenizerProvider,
    build_hf_tokenizer_handle_from_model,
)
from veeksha.new.generator.session.registry import SessionGeneratorRegistry
from veeksha.new.types import ChannelModality

logger = init_logger(__name__)


def setup_api_environment(
    api_key=None,
    api_url=None,
):
    """Set up environment variables for OpenAI API"""
    assert api_key is not None, "API key is required"
    assert api_url is not None, "API URL is required"
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_url


def _send_probe_request(
    url: str, headers: dict, body: dict, min_param: str, param_value
) -> bool:
    """Send a probe request to test if server accepts a parameter."""
    import requests  # type: ignore

    test_body = body.copy()
    test_body[min_param] = param_value
    try:
        resp = requests.post(url, headers=headers, json=test_body, timeout=10)
        return 200 <= resp.status_code < 300
    except Exception:
        return False


def _probe_min_tokens_param_support(client_config: ClientConfig) -> bool:
    """Probe if server accepts the configured min token parameter."""
    import os

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
        "max_completion_tokens": 1,
    }
    if client_config.llm_api == "openai_completions":
        body["prompt"] = "Hello"
    else:
        body["messages"] = [{"role": "user", "content": "Hello"}]

    if not _send_probe_request(url, headers, body, min_param, 1):
        logger.warning(
            f"Server rejected parameter '{min_param}'; falling back to prompt control."
        )
        return False

    if not _send_probe_request(url, headers, body, min_param, {"invalid": "type"}):
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
    # Worker input queues; 1 per worker thread
    input_queues = [Queue() for _ in range(benchmark_config.num_request_runner_threads)]
    output_queue = Queue()  # Worker output queue
    ready_queue = Queue()  # Prefetch -> Dispatcher queue
    stop_event = threading.Event()
    scheduler = DispatchScheduler()

    # Create benchmark context
    benchmark_context = BenchmarkContext(
        benchmark_id=benchmark_id,
        telemetry_enabled=benchmark_config.runtime_telemetry_enabled,
    )

    # Initialize request runner
    req_runner = RequestRunnerManager(
        client_config=benchmark_config.client_config,
        input_queues=input_queues,
        output_queue=output_queue,
        num_threads=benchmark_config.num_request_runner_threads,
    )

    # Start the worker threads
    req_runner.start()

    # Create locks for thread-safe access to shared resources
    generator_lock = threading.Lock()  # Protects request generator
    responses_lock = threading.Lock()  # Protects generated_responses list
    pbar_lock = threading.Lock()  # Protects progress bar

    # Create thread pool manager
    pool_manager = ThreadPoolManager(stop_event=stop_event)

    # Create prefetch worker pool
    pool_manager.create_pool(
        name="prefetch",
        worker_class=PrefetchWorker,
        worker_kwargs={
            "ready_queue": ready_queue,
            "service_metrics": service_metrics,
            "request_generator": request_generator,
            "generator_lock": generator_lock,
        },
        pool_size=benchmark_config.num_prefetch_threads,
    )

    # Create dispatcher worker pool
    pool_manager.create_pool(
        name="dispatcher",
        worker_class=DispatchWorker,
        worker_kwargs={
            "input_queues": input_queues,
            "ready_queue": ready_queue,
            "service_metrics": service_metrics,
            "scheduler": scheduler,
            "req_runner": req_runner,
            "benchmark_context": benchmark_context,
        },
        pool_size=benchmark_config.num_dispatcher_threads,
    )

    # Create results processor worker pool
    pool_manager.create_pool(
        name="processor",
        worker_class=ResultsProcessorWorker,
        worker_kwargs={
            "output_queue": output_queue,
            "service_metrics": service_metrics,
            "generated_responses": generated_responses,
            "responses_lock": responses_lock,
            "pbar": pbar,
            "pbar_lock": pbar_lock,
            "scheduler": scheduler,
        },
        pool_size=benchmark_config.num_results_processor_threads,
    )

    # Start all thread pools
    pool_manager.start_all()

    logger.info(
        f"Started {pool_manager.get_total_thread_count()} threads across thread pools "
        f"and {req_runner.get_worker_count()} worker threads"
    )

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

    # Wait for prefetch threads to finish
    pool_manager.join_pool("prefetch")

    # Wait for dispatcher threads to finish
    pool_manager.join_pool("dispatcher")

    # Wait for all worker threads to terminate
    req_runner.wait_for_workers()
    logger.debug("Worker threads joined")

    # Signal the results processor threads to finish after draining and join them
    num_processor_threads = benchmark_config.num_results_processor_threads
    for _ in range(num_processor_threads):
        output_queue.put(None)  # One sentinel per processor thread
    pool_manager.join_pool("processor")

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

    logger.info(
        "Running benchmark with config:\n%s",
        benchmark_config,
    )

    # 0. Prepare benchmark
    #   - Prepare output directory
    #   - Set seed
    #   - Set environment variables
    #   - WandB
    seed_manager = SeedManager(benchmark_config.seed)
    # 1. Get content generator
    logger.info(
        f"Session generator type: {benchmark_config.session_generator.get_type()}"
    )
    tokenizer_provider = TokenizerProvider(
        {
            ChannelModality.TEXT: build_hf_tokenizer_handle_from_model(
                benchmark_config.model
            )
        }
    )
    session_generator = SessionGeneratorRegistry.get(
        benchmark_config.session_generator.get_type(),
        config=benchmark_config.session_generator,
        seed_manager=seed_manager,
        tokenizer_provider=tokenizer_provider,
    )
    logger.info(f"Session generator: {session_generator}")
    session = session_generator.generate_session()
    print_session(session)
    print(session.requests[0].channels[ChannelModality.TEXT])
    print(session.requests[1].channels[ChannelModality.TEXT])
    # 2. Create dispatchers and request runners
    # 3. Get evaluator (metrics collector)
    # 4. Run the benchmark
    # 5. Flush final metrics

    # prepare_benchmark_output_dir(benchmark_config)

    # random.seed(benchmark_config.seed)

    # Generate unique benchmark ID from output directory
    # benchmark_id = os.path.basename(benchmark_config.metrics_config.output_dir)
    # logger.info(
    #     f"Benchmark ID: {benchmark_id}, Output directory: {benchmark_config.metrics_config.output_dir}"
    # )
    # # Expose output directory to child threads for iterative trace writes
    # os.environ["VEEKSHA_OUTPUT_DIR"] = benchmark_config.metrics_config.output_dir
    # os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # if benchmark_config.server_config is None:
    #     setup_api_environment(
    #         api_key=benchmark_config.api_key,
    #         api_url=benchmark_config.api_url,
    #     )
    # else:
    #     logger.info("Using API environment from managed server configuration")

    # TODO: valid only for if text channel is present
    # _initialize_min_tokens_support(benchmark_config) # TODO: Uncomment this when we have a server that supports it
    # tokenizer = get_tokenizer(
    #     tokenizer_name=benchmark_config.client_config.tokenizer, # type: ignore
    #     trust_remote_code=True,
    # )
    # generated_responses: List[Response] = []

    # request_generator_params = {}
    # request_generator_config_type = benchmark_config.request_generator_config.get_type()

    # if (
    #     request_generator_config_type == RequestGeneratorType.SYNTHETIC
    #     or request_generator_config_type == RequestGeneratorType.TRACE
    # ):
    #     request_generator_params = {
    #         "corpus_lines": load_corpus(),
    #     }

    # request_generator = RequestGeneratorRegistry.get(
    #     benchmark_config.request_generator_config.get_type(),
    #     config=benchmark_config.request_generator_config,
    #     tokenizer=tokenizer,
    #     client_config=benchmark_config.client_config,
    #     seed_manager=seed_manager,
    #     **request_generator_params,
    # )

    # max_requests = (
    #     request_generator.num_requests
    #     if benchmark_config.request_generator_config.get_type()
    #     == RequestGeneratorType.LMEVAL
    #     else benchmark_config.max_completed_requests
    # )
    # # Disable tqdm progress bar if dashboard is enabled to prevent output conflicts
    # pbar = tqdm(total=max_requests, disable=benchmark_config.dashboard_config.enabled)

    # service_metrics = ServiceMetrics(
    #     max_requests=max_requests,
    #     timeout=benchmark_config.timeout,
    #     metrics_config=benchmark_config.metrics_config,
    # )

    # run_main_loop(
    #     benchmark_config=benchmark_config,
    #     request_generator=request_generator,
    #     service_metrics=service_metrics,
    #     generated_responses=generated_responses,
    #     pbar=pbar,
    #     benchmark_id=benchmark_id,
    # )

    # service_metrics.store_output()
    # logger.info(f"Metrics stored to {service_metrics.output_dir}")

    # store_generated_texts(service_metrics.output_dir, generated_responses)

    # lm-eval specific
    # if (
    #     benchmark_config.request_generator_config.get_type()
    #     == RequestGeneratorType.LMEVAL
    # ):
    #     request_generator.get_responses(generated_responses)
    #     lmeval_results = request_generator.evaluate()

    #     store_lmeval_results(service_metrics.output_dir, lmeval_results)

    return None


def run_benchmarks(benchmark_configs: List[BenchmarkConfig]) -> List[ServiceMetrics]:
    """Run benchmark with console-only output

    Args:
        benchmark_configs: List of configurations for the benchmarks
    """
    service_metrics_list: List[ServiceMetrics] = []
    if len(benchmark_configs) > 1:
        logger.info(
            f"Running {len(benchmark_configs)} benchmark configurations sequentially."
        )

    for i, benchmark_config in enumerate(benchmark_configs):
        print(f"Running benchmark with config: {benchmark_config}")
        if len(benchmark_configs) > 1:
            logger.info(f"Starting benchmark {i+1}/{len(benchmark_configs)}")

        is_last = i == len(benchmark_configs) - 1

        if benchmark_config.server_config is not None:
            logger.info("Server configuration detected - using managed server")
            from veeksha.orchestration import managed_server

            prepare_benchmark_output_dir(benchmark_config)
            os.environ["VEEKSHA_OUTPUT_DIR"] = (
                benchmark_config.metrics_config.output_dir
            )

            logger.info(f"Launching {benchmark_config.server_config.engine} server...")
            with managed_server(benchmark_config.server_config) as info:
                logger.info(f"Server ready at {info['api_base']}")
                logger.info("Running benchmark...")
                service_metrics_list.append(
                    run_benchmark(benchmark_config=benchmark_config)
                )
            logger.info("Server shut down")
        else:
            # assume external server is running
            service_metrics_list.append(
                run_benchmark(benchmark_config=benchmark_config)
            )

        if len(benchmark_configs) > 1:
            logger.info(f"Completed benchmark {i+1}/{len(benchmark_configs)}")

        logger.info("All benchmarks completed.")
    return service_metrics_list


if __name__ == "__main__":
    pass
