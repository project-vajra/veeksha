import multiprocessing
import os
import platform
import random
import shutil
import threading
import time
import hashlib
import json
from multiprocessing import Queue
from queue import Empty
from threading import Thread
from typing import List, Dict, Any

from tqdm import tqdm  # type: ignore

from veeksha.benchmark_data_utils import (
    load_corpus,
    store_generated_texts,
    store_lmeval_results,
)
from veeksha.config.config import BenchmarkConfig
from veeksha.core.hf_utils import get_tokenizer
from veeksha.core.requests_launcher import RequestsLauncher
from veeksha.core.response import Response
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics
from veeksha.request_generator.base_generator import BaseRequestGenerator
from veeksha.request_generator.interval_generator.base_generator import (
    BaseRequestIntervalGenerator,
)
from veeksha.request_generator.interval_generator.generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from veeksha.request_generator.request_generator_registry import (
    RequestGeneratorRegistry,
)
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
    requests_interval_generator: BaseRequestIntervalGenerator,
    request_generator: BaseRequestGenerator,
    stop_event: threading.Event,
) -> None:
    """Thread function to generate and dispatch requests."""
    print("Starting dispatch_requests thread")
    num_errored_requests_handled = 0
    request_count = 0

    while not stop_event.is_set():
        if should_send_new_request(service_metrics, num_errored_requests_handled):
            request_start_time = time.monotonic()
            request_count += 1
            
            if request_count % 10 == 0:
                print(f"Dispatching request #{request_count}")

            # Check if we should handle error request
            if service_metrics.num_requests >= service_metrics.max_requests:
                num_errored_requests_handled += 1
                print(f"Handling error request #{num_errored_requests_handled}")

            # Create and dispatch request
            print(f"Registering launched request #{service_metrics.num_requests + 1}")
            service_metrics.register_launched_request()
            print("Getting request configuration")
            request_config = request_generator.get_request()
            print(f"Putting request in input queue (queue size approx: {input_queue.qsize() if hasattr(input_queue, 'qsize') else 'unknown'})")
            input_queue.put(request_config)

            # Wait for next interval
            print("Getting next request interval")
            next_request_interval = (
                requests_interval_generator.get_next_inter_request_time()
            )

            if next_request_interval < 0:
                print(f"Invalid interval {next_request_interval}, stopping main loop")
                logger.warning(
                    f"Invalid interval {next_request_interval} (potentially from trace interval generator). Stopping the main loop."
                )
                break

            print(f"Waiting for {next_request_interval:.4f}s before next request")
            while not stop_event.is_set():
                if time.monotonic() - request_start_time >= next_request_interval:
                    break
                time.sleep(next_request_interval)
        else:
            time.sleep(0.005)
    
    print("Exiting dispatch_requests thread")


def process_results(
    output_queue: Queue,
    service_metrics: ServiceMetrics,
    generated_responses: List[Response],
    pbar: tqdm,
    stop_event: threading.Event,
) -> None:
    """Thread function to process results from the output queue."""
    print("Starting process_results thread")
    result_count = 0
    
    while not stop_event.is_set() or not output_queue.empty():
        try:
            print(f"Trying to get result from output queue (queue size approx: {output_queue.qsize() if hasattr(output_queue, 'qsize') else 'unknown'})")
            result = output_queue.get(timeout=0.1)
            result_count += 1
            print(f"Got result #{result_count} from output queue")
            
            request_metrics, generated_response = result
            if generated_response:
                print(f"Adding metrics for completed request #{service_metrics.num_completed_requests + 1}")
                service_metrics.add_request_metrics(request_metrics)
                generated_responses.append(generated_response)
                print(f"Total completed responses: {len(generated_responses)}")
            else:
                print("Received result with no generated response")

            pbar.update(service_metrics.num_completed_requests - pbar.n)
            
            if result_count % 10 == 0:
                print(f"Processed {result_count} results. Completed: {service_metrics.num_completed_requests}, Errors: {service_metrics.num_errored_requests}")
        except Empty:
            if result_count % 100 == 0:  # Only print occasionally to avoid log spam
                print("Output queue empty, waiting for more results...")
            continue
    
    print("Exiting process_results thread")


def run_main_loop(
    benchmark_config: BenchmarkConfig,
    requests_interval_generator: BaseRequestIntervalGenerator,
    request_generator: BaseRequestGenerator,
    service_metrics: ServiceMetrics,
    generated_responses: List[Response],
    pbar: tqdm,
):
    """Run the main loop for the benchmark."""

    logger.info("Starting the main loop.")
    print("Starting the main loop")

    # Create queues for communication
    print("Creating communication queues")
    input_queue = Queue()
    output_queue = Queue()
    stop_event = threading.Event()

    # Initialize request launcher
    print("Initializing request launcher")
    req_launcher = RequestsLauncher(
        client_config=benchmark_config.client_config,
        input_queue=input_queue,
        output_queue=output_queue,
    )

    # Start the request launcher processes
    print("Starting request launcher processes")
    req_launcher.start()
    print("Request launcher processes started")

    # Create and start producer-consumer threads
    print("Creating dispatcher thread")
    dispatcher_thread = Thread(
        target=dispatch_requests,
        args=(
            input_queue,
            service_metrics,
            requests_interval_generator,
            request_generator,
            stop_event,
        ),
    )

    print("Creating processor thread")
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

    print("Starting dispatcher thread")
    dispatcher_thread.start()
    print("Starting processor thread")
    processor_thread.start()
    print("Both threads started")

    # Monitor and wait for completion
    print("Entering monitoring loop with service_metrics")
    with service_metrics:
        print("Inside service_metrics context manager")
        counter = 0
        while not service_metrics.should_stop():
            time.sleep(0.1)
            counter += 1
            if counter % 100 == 0:  # Print every 10 seconds
                print(f"Still in monitoring loop. Requests: {service_metrics.num_requests}/{service_metrics.max_requests}, Completed: {service_metrics.num_completed_requests}, Errors: {service_metrics.num_errored_requests}")
        print("Exiting monitoring loop - service_metrics indicated we should stop")
        logger.info("Stopping the main loop.")

    # Signal threads to stop and wait for completion
    print("Setting stop event to terminate threads")
    stop_event.set()
    print("Waiting for dispatcher thread to join")
    dispatcher_thread.join()
    print("Waiting for processor thread to join")
    processor_thread.join()
    print("Both threads joined")

    # Terminate all clients
    print("Terminating request launcher clients")
    req_launcher.kill_clients()
    print("All clients terminated")

    pbar.close()
    print("Progress bar closed")
    logger.info("Main loop completed.")
    print("Main loop completed")


# Generate a hash of the benchmark configuration to use for caching
def get_config_hash(config: BenchmarkConfig) -> str:
    config_str = json.dumps(config.to_dict(), sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


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
    
    print("Starting run_benchmark function")
    # Check if this benchmark has already been run
    config_hash = get_config_hash(benchmark_config)
    output_dir = benchmark_config.metrics_config.output_dir
    cache_marker_file = os.path.join(output_dir, f".completed_{config_hash}")
    
    # If the cache marker file exists, skip this benchmark run
    if os.path.exists(cache_marker_file):
        logger.info(f"Skipping benchmark as it was already completed (hash: {config_hash})")
        return
    
    print("Benchmark not found in cache, proceeding with execution")    
    generated_responses: List[Response] = []

    print("Setting up request interval generator")
    requests_interval_generator = RequestIntervalGeneratorRegistry.get(
        benchmark_config.request_interval_generator_config.get_type(),
        benchmark_config.request_interval_generator_config,
    )

    assert (
        benchmark_config.client_config.tokenizer is not None
    ), "Tokenizer is required."

    print("Loading tokenizer")
    tokenizer = get_tokenizer(
        tokenizer_name=benchmark_config.client_config.tokenizer,
        trust_remote_code=True,
    )

    request_generator_params = {}

    if (
        benchmark_config.request_generator_config.get_type()
        == RequestGeneratorType.SYNTHETIC
    ):
        print("Loading corpus for synthetic request generator")
        request_generator_params = {
            "corpus_lines": load_corpus(),
        }

    print("Setting up request generator")
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
    print(f"Setting up progress bar with max_requests={max_requests}")
    pbar = tqdm(total=max_requests)

    print("Initializing service metrics")
    service_metrics = ServiceMetrics(
        max_requests=max_requests,
        timeout=benchmark_config.timeout,
        deadline_config=benchmark_config.deadline_config,
        metrics_config=benchmark_config.metrics_config,
        prefill_profiler_config=benchmark_config.prefill_profiler_config,
    )

    print("About to enter main loop")
    run_main_loop(
        benchmark_config=benchmark_config,
        requests_interval_generator=requests_interval_generator,
        request_generator=request_generator,
        service_metrics=service_metrics,
        generated_responses=generated_responses,
        pbar=pbar,
    )
    print("Finished main loop")

    logger.info(
        f"Results for token benchmark for {benchmark_config.client_config.model} queried with the {benchmark_config.client_config.llm_api} api. {service_metrics}"
    )

    print("Storing service metrics")
    service_metrics.store_output()
    logger.info(f"Metrics stored to {service_metrics.output_dir}")

    print("Storing generated texts")
    store_generated_texts(service_metrics.output_dir, generated_responses)

    # lm-eval specific
    if (
        benchmark_config.request_generator_config.get_type()
        == RequestGeneratorType.LMEVAL
    ):
        print("Processing lm-eval results")
        request_generator.get_responses(generated_responses)
        lmeval_results = request_generator.evaluate()
        logger.info(f"Results: {lmeval_results}")

        store_lmeval_results(service_metrics.output_dir, lmeval_results)
    
    print("Creating cache marker file")
    # Create a marker file to indicate this benchmark run was completed successfully
    with open(cache_marker_file, "w") as f:
        f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config hash: {config_hash}\n")
    logger.info(f"Created cache marker file: {cache_marker_file}")
    print("Benchmark completed successfully")


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    benchmark_config: BenchmarkConfig = BenchmarkConfig.create_from_cli_args()
    random.seed(benchmark_config.seed)

    # setup output directory
    # delete output directory if exists
    if os.path.exists(benchmark_config.metrics_config.output_dir):
        shutil.rmtree(benchmark_config.metrics_config.output_dir)
    
    os.makedirs(benchmark_config.metrics_config.output_dir, exist_ok=True)

    run_benchmark(benchmark_config=benchmark_config)
