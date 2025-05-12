import asyncio
import multiprocessing
import platform
import random
import time
import os
from multiprocessing import Queue
from queue import Empty
from typing import List, Dict, Tuple

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
from veeksha.metrics.request_metrics import RequestMetrics
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


async def dispatch_requests(
    input_queue: Queue,
    service_metrics: ServiceMetrics,
    requests_interval_generator: BaseRequestIntervalGenerator,
    request_generator: BaseRequestGenerator,
    stop_event: asyncio.Event,
) -> None:
    """Async function to generate and dispatch requests."""
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
            await asyncio.to_thread(input_queue.put, request_config)

            # Wait for next interval
            next_request_interval = (
                requests_interval_generator.get_next_inter_request_time()
            )

            if next_request_interval < 0:
                logger.warning(
                    f"Invalid interval {next_request_interval} (potentially from trace interval generator). Stopping the main loop."
                )
                break

            # Wait for the next request interval using asyncio.sleep
            remainder = next_request_interval
            while not stop_event.is_set() and remainder > 0:
                sleep_time = min(remainder, 0.01)
                await asyncio.sleep(sleep_time)
                remainder = next_request_interval - (time.monotonic() - request_start_time)
        else:
            await asyncio.sleep(0.01)


async def process_results(
    output_queue: Queue,
    service_metrics: ServiceMetrics,
    generated_responses: List[Response],
    pbar: tqdm,
    stop_event: asyncio.Event,
) -> None:
    """Async function to process results from the output queue."""
    while not stop_event.is_set() or not output_queue.empty():
        try:
            # Use run_in_executor to perform blocking queue.get
            result = await asyncio.to_thread(output_queue.get, True, 0.1)
            request_metrics, generated_response = result
            if generated_response:
                service_metrics.add_request_metrics(request_metrics)
                generated_responses.append(generated_response)

            pbar.update(service_metrics.num_completed_requests - pbar.n)
        except Empty:
            await asyncio.sleep(0.01)


async def run_main_loop(
    benchmark_config: BenchmarkConfig,
    requests_interval_generator: BaseRequestIntervalGenerator,
    request_generator: BaseRequestGenerator,
    service_metrics: ServiceMetrics,
    generated_responses: List[Response],
    pbar: tqdm,
):
    """Run the main loop for the benchmark using asyncio."""

    print("Starting the main loop.")

    # Create queues for communication
    input_queue = Queue()
    output_queue = Queue()
    stop_event = asyncio.Event()

    # Initialize request launcher
    req_launcher = RequestsLauncher(
        client_config=benchmark_config.client_config,
        input_queue=input_queue,
        output_queue=output_queue,
        total_benchmark_time=service_metrics.timeout,
    )

    # Start the request launcher processes
    req_launcher.start()

    # Create and start tasks
    dispatcher_task = asyncio.create_task(
        dispatch_requests(
            input_queue,
            service_metrics,
            requests_interval_generator,
            request_generator,
            stop_event,
        )
    )

    processor_task = asyncio.create_task(
        process_results(
            output_queue,
            service_metrics,
            generated_responses,
            pbar,
            stop_event,
        )
    )

    # Monitor and wait for completion
    with service_metrics:
        while not service_metrics.should_stop():
            await asyncio.sleep(0.1)
        print("Stopping the main loop.")

    # Signal tasks to stop 
    print("Setting stop event to terminate tasks")
    stop_event.set()
    
    print("Waiting for dispatcher task to complete...")
    await dispatcher_task
    print("Dispatcher task terminated")
    
    print("Waiting for processor task to complete...")
    await processor_task
    print("Processor task terminated")

    # Terminate all clients
    print("Terminating all client connections")
    
    # Get the final state of unfinished requests
    # Try to sleep briefly to ensure all debug output has been printed
    await asyncio.sleep(1)
    
    print("\n--------- FINAL REQUEST METRICS ---------")
    
    unfinished_count = req_launcher.get_unfinished_requests_count()
    print(f"Final request state: {unfinished_count} unfinished")
    
    num_reqs_without_prefill = 0
    # Keep track of any unfinished requests for reporting
    if unfinished_count > 0:
        print("\nUnfinished requests by client ID:")
        unfinished_requests = req_launcher.get_unfinished_requests()
        for client_id, metrics_list in unfinished_requests.items():
            if metrics_list:  # Check if the list is not empty
                print(f"  Client {client_id} has {len(metrics_list)} unfinished requests")
                for metrics, response in metrics_list:
                    # Add metrics to the service metrics
                    if metrics.num_output_tokens > 0:
                        service_metrics.add_request_metrics(metrics)
                    else:
                        num_reqs_without_prefill += 1

    print(f"Out of {unfinished_count} unfinished requests, {num_reqs_without_prefill} requests didn't finish prefill.")
    
    req_launcher.kill_clients()
    print("All clients terminated")

    pbar.close()
    print("Benchmark run completed. All resources cleaned up.")
    print("Main loop completed.")


async def run_benchmark(
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

    await run_main_loop(
        benchmark_config=benchmark_config,
        requests_interval_generator=requests_interval_generator,
        request_generator=request_generator,
        service_metrics=service_metrics,
        generated_responses=generated_responses,
        pbar=pbar,
    )

    print(
        f"Results for token benchmark for {benchmark_config.client_config.model} queried with the {benchmark_config.client_config.llm_api} api. {service_metrics}"
    )

    service_metrics.store_output()
    print(f"Metrics stored to {service_metrics.output_dir}")

    store_generated_texts(service_metrics.output_dir, generated_responses)

    # lm-eval specific
    if (
        benchmark_config.request_generator_config.get_type()
        == RequestGeneratorType.LMEVAL
    ):
        request_generator.get_responses(generated_responses)
        lmeval_results = request_generator.evaluate()
        print(f"Results: {lmeval_results}")

        store_lmeval_results(service_metrics.output_dir, lmeval_results)


def main():
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    benchmark_config: BenchmarkConfig = BenchmarkConfig.create_from_cli_args()
    random.seed(benchmark_config.seed)
    asyncio.run(run_benchmark(benchmark_config=benchmark_config))
    os._exit(0)

if __name__ == "__main__":
    main()
