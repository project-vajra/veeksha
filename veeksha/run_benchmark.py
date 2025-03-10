import json
import multiprocessing
import os
import platform
import random
import threading
import time
from multiprocessing import Queue
from queue import Empty
from threading import Thread
from typing import List

from tqdm import tqdm  # type: ignore

from veeksha.config.config import BenchmarkConfig
from veeksha.core.hf_utils import get_tokenizer
from veeksha.core.requests_launcher import RequestsLauncher
from veeksha.core.response import Response
from veeksha.logger import init_logger
from veeksha.metrics.service_metrics import ServiceMetrics
from veeksha.request_generator.base_generator import BaseRequestGenerator
from veeksha.request_generator.generator_registry import RequestGeneratorRegistry
from veeksha.request_generator.interval_generator.base_generator import (
    BaseRequestIntervalGenerator,
)
from veeksha.request_generator.interval_generator.generator_registry import (
    RequestIntervalGeneratorRegistry,
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
    num_errored_requests_handled = 0

    while not stop_event.is_set():
        if should_send_new_request(service_metrics, num_errored_requests_handled):
            request_start_time = time.monotonic()

            # Check if we should handle error request
            if service_metrics.num_requests >= service_metrics.max_requests:
                num_errored_requests_handled += 1

            # Create and dispatch request
            service_metrics.register_launched_request()
            request_config = request_generator.get_request()
            input_queue.put(request_config)

            # Wait for next interval
            next_request_interval = (
                requests_interval_generator.get_next_inter_request_time()
            )

            if next_request_interval < 0:
                logger.warning(
                    f"Invalid interval {next_request_interval} (potentially from trace interval generator). Stopping the main loop."
                )
                break

            while not stop_event.is_set():
                if time.monotonic() - request_start_time >= next_request_interval:
                    break
                time.sleep(0.01)
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
    requests_interval_generator: BaseRequestIntervalGenerator,
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
            requests_interval_generator,
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

    requests_interval_generator = RequestIntervalGeneratorRegistry.get(
        benchmark_config.request_interval_generator_config.get_type(),
        benchmark_config.request_interval_generator_config,
    )

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
        corpus_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
        )
        with open(corpus_path, "r") as f:
            corpus_lines = f.readlines()
        request_generator_params = {
            "corpus_lines": corpus_lines,
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
        requests_interval_generator=requests_interval_generator,
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

    # store the generated texts
    with open(
        os.path.join(service_metrics.output_dir, "generated_texts.txt"), "w"
    ) as f:
        f.write(("\n" + "-" * 30 + "\n").join([i.text for i in generated_responses]))

    # lm-eval specific
    if (
        benchmark_config.request_generator_config.get_type()
        == RequestGeneratorType.LMEVAL
    ):
        request_generator.get_responses(generated_responses)
        lmeval_results = request_generator.evaluate()
        logger.info(f"Results: {lmeval_results}")

        # write results dictionary to file
        with open(
            os.path.join(service_metrics.output_dir, "lmeval_results.json"), "w"
        ) as f:
            json.dump(lmeval_results, f, indent=4)


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    benchmark_config: BenchmarkConfig = BenchmarkConfig.create_from_cli_args()
    random.seed(benchmark_config.seed)
    run_benchmark(benchmark_config=benchmark_config)
