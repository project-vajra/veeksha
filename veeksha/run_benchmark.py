import multiprocessing
import platform
import random
import threading
import time
from multiprocessing import Queue
from queue import Empty
from threading import Thread
from typing import List

from tqdm import tqdm  # type: ignore

from veeksha.benchmark_data_utils import (
    load_corpus,
    store_generated_texts,
    store_lmeval_results,
)
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.core.hf_utils import get_tokenizer
from veeksha.core.requests_launcher import RequestsLauncher
from veeksha.core.response import Response
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.generators.request_generator.generator_registry import (
    RequestGeneratorRegistry,
)
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics
from veeksha.types import RequestGeneratorType

logger = init_logger(__name__)


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
) -> None:
    """Thread function to generate and dispatch requests."""
    num_errored_requests_handled = 0

    while not stop_event.is_set():
        if should_send_new_request(service_metrics, num_errored_requests_handled):
            request_start_time = time.monotonic()

            # Check if we should handle error request
            if service_metrics.num_requests >= service_metrics.max_requests:
                num_errored_requests_handled += 1

            # Get next request and its dispatch time
            request_config = request_generator.get_request()
            request_dispatch_interval = request_config.metadata[
                "request_dispatch_interval"
            ]
            service_metrics.register_launched_request()

            if request_dispatch_interval < 0:
                logger.warning(
                    f"Invalid interval {request_dispatch_interval} (potentially from trace interval generator). Stopping the main loop."
                )
                break

            # Wait for dispatch time
            while not stop_event.is_set():
                elapsed_time = time.monotonic() - request_start_time
                if elapsed_time >= request_dispatch_interval:
                    break
                # remaining sleep time to avoid drift
                remaining_time = request_dispatch_interval - elapsed_time
                if remaining_time > 0:
                    # capped sleep at 100ms
                    sleep_duration = min(remaining_time, 0.1)
                    time.sleep(sleep_duration)

            # Dispatch request
            input_queue.put(request_config)
            logger.info(
                f"Dispatched request {request_config.id}: {request_config.metadata['input_length']} prefill, {request_config.metadata['output_length']} decode"
            )
        else:
            time.sleep(0.01)


def process_results(
    output_queue: Queue,
    service_metrics: ServiceMetrics,
    generated_responses: List[Response],
    pbar: tqdm,
    stop_event: threading.Event,
) -> None:
    """Thread function to process results from the output queue."""
    while not stop_event.is_set() or not output_queue.empty():
        try:
            result = output_queue.get(timeout=0.1)
            request_metrics, generated_response = result
            if generated_response:
                service_metrics.add_request_metrics(request_metrics)
                generated_responses.append(generated_response)

            pbar.update(service_metrics.num_completed_requests - pbar.n)
        except Empty:
            continue


def run_main_loop(
    benchmark_config: BenchmarkConfig,
    request_generator: BaseRequestGenerator,
    service_metrics: ServiceMetrics,
    generated_responses: List[Response],
    pbar: tqdm,
):
    """Run the main loop for the benchmark."""

    logger.info("Starting the main loop.")

    # Create queues for communication
    input_queue = Queue()
    output_queue = Queue()
    stop_event = threading.Event()

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
        ),
    )

    dispatcher_thread.start()
    processor_thread.start()

    # Monitor and wait for completion
    with service_metrics:
        while not service_metrics.should_stop():
            time.sleep(0.1)
        logger.info("Stopping the main loop.")

    # Signal threads to stop and wait for completion
    stop_event.set()
    dispatcher_thread.join()
    processor_thread.join()

    # Terminate all clients
    req_launcher.kill_clients()

    pbar.close()
    logger.info("Main loop completed.")


def run_benchmark(
    benchmark_config: BenchmarkConfig,
):
    """Get the token throughput and latencies for the given model.

    Args:
        benchmark_config: The benchmark configuration.

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """

    generated_responses: List[Response] = []

    assert (
        benchmark_config.client_config.tokenizer is not None
    ), "Tokenizer is required."

    tokenizer = get_tokenizer(
        tokenizer_name=benchmark_config.client_config.tokenizer,
        trust_remote_code=True,
    )

    request_generator_params = {}

    if (
        benchmark_config.request_generator_config.get_type()
        == RequestGeneratorType.SYNTHETIC
    ):
        request_generator_params = {
            "corpus_lines": load_corpus(),
        }

    request_generator = RequestGeneratorRegistry.get(
        benchmark_config.request_generator_config.get_type(),
        config=benchmark_config.request_generator_config,
        tokenizer=tokenizer,
        client_config=benchmark_config.client_config,
        **request_generator_params,
    )

    max_requests = (
        request_generator.num_requests
        if benchmark_config.request_generator_config.get_type()
        == RequestGeneratorType.LMEVAL
        else benchmark_config.max_completed_requests
    )
    pbar = tqdm(total=max_requests)

    service_metrics = ServiceMetrics(
        max_requests=max_requests,
        timeout=benchmark_config.timeout,
        deadline_config=benchmark_config.deadline_config,
        metrics_config=benchmark_config.metrics_config,
        prefill_profiler_config=benchmark_config.prefill_profiler_config,
    )

    run_main_loop(
        benchmark_config=benchmark_config,
        request_generator=request_generator,
        service_metrics=service_metrics,
        generated_responses=generated_responses,
        pbar=pbar,
    )

    logger.info(
        f"Results for token benchmark for {benchmark_config.client_config.model} queried with the {benchmark_config.client_config.llm_api} api. {service_metrics}"
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
        logger.info(f"Results: {lmeval_results}")

        store_lmeval_results(service_metrics.output_dir, lmeval_results)


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    benchmark_configs = BenchmarkConfig.create_from_cli_args()
    
    if len(benchmark_configs) > 1:
        logger.info(f"Running {len(benchmark_configs)} benchmark configurations sequentially.")
    
    for i, benchmark_config in enumerate(benchmark_configs):
        if len(benchmark_configs) > 1:
            logger.info(f"Starting benchmark {i+1}/{len(benchmark_configs)}")
        
        random.seed(benchmark_config.seed)
        run_benchmark(benchmark_config=benchmark_config)
        
        if len(benchmark_configs) > 1:
            logger.info(f"Completed benchmark {i+1}/{len(benchmark_configs)}")
    
    logger.info("All benchmarks completed.")
