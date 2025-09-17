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

from veeksha.file_utils import (
    store_generated_texts,
    store_lmeval_results,
)
from veeksha.config.benchmark_config import BenchmarkConfig
from veeksha.core.hf_utils import get_tokenizer
from veeksha.core.requests_launcher import RequestsLauncher
from veeksha.core.response import Response
from veeksha.generators.request_generator.base_request_generator import BaseRequestGenerator
from veeksha.generators.request_generator.request_generator_registry import (
    RequestGeneratorRegistry,
)
from veeksha.logger import init_logger
from veeksha.benchmark_tracker import BenchmarkTracker
from veeksha.types import RequestGeneratorType

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


def should_send_new_request(
    benchmark_tracker: BenchmarkTracker, num_errored_requests_handled: int
) -> bool:
    """Check if a request should be sent based on the current state of the service."""
    return (benchmark_tracker.num_requests < benchmark_tracker.max_requests) or (
        benchmark_tracker.num_requests >= benchmark_tracker.max_requests
        and num_errored_requests_handled < benchmark_tracker.num_errored_requests
    )


def build_unique_output_dir(root: str, model_name: str, config_hash: str) -> str:
    """Return a unique timestamped output directory path.

    Format: <root>/<model>-<hash>-<timestamp>
    """
    timestamp = (
        time.strftime("%Y%m%d-%H%M%S", time.localtime())
        + f"-{int(time.time()*1000)%1000:03d}"
    )
    return os.path.join(root, f"{model_name}-{config_hash}-{timestamp}")


def prepare_benchmark_output_dir(benchmark_config: BenchmarkConfig) -> None:
    """Create a unique output subdirectory and persist config.
    - Create a unique subdirectory under `output_dir`,
      named with model and config-hash plus a high-entropy timestamp.
    - Save `config.json` in the final output directory.
    """

    base_output_dir = benchmark_config.output_dir
    model_name = benchmark_config.client_config.model.split("/")[-1]

    config_hash = benchmark_config.get_hash()
    unique_dir = build_unique_output_dir(base_output_dir, model_name, config_hash)
    object.__setattr__(benchmark_config, "output_dir", unique_dir)
    benchmark_config.write_config_to_file()


def dispatch_requests(
    input_queue: Queue,
    benchmark_tracker: BenchmarkTracker,
    request_generator: BaseRequestGenerator,
    stop_event: threading.Event,
) -> None:
    """Thread function to generate and dispatch requests."""
    num_errored_requests_handled = 0

    while not stop_event.is_set():
        if should_send_new_request(benchmark_tracker, num_errored_requests_handled):
            request_start_time = time.monotonic()

            # check if we should handle error request
            if benchmark_tracker.num_requests >= benchmark_tracker.max_requests:
                num_errored_requests_handled += 1

            # get next request and its dispatch time
            try:
                request_config = request_generator.get_request()
            except StopIteration as e:
                benchmark_tracker.notify_error(e)
                stop_event.set()
                break
            request_dispatch_delay = request_config.dispatch_delay

            if request_dispatch_delay == -1:
                logger.info(
                    "Benchmark ending early due to stop policy (generator sentinel received)."
                )
                benchmark_tracker.request_stop()
                stop_event.set()
                break
            elif request_dispatch_delay < 0:
                raise ValueError(
                    f"Invalid request dispatch delay '{request_dispatch_delay}' from request metadata."
                )

            # wait for dispatch time
            while not stop_event.is_set():
                elapsed_time = time.monotonic() - request_start_time
                if elapsed_time >= request_dispatch_delay:
                    break
                # remaining sleep time to avoid drift
                remaining_time = request_dispatch_delay - elapsed_time
                if remaining_time > 0:
                    # capped sleep at 100ms
                    sleep_duration = min(remaining_time, 0.1)
                    time.sleep(sleep_duration)

            # if another thread has set the stop event we don't send the request
            if stop_event.is_set():
                continue

            # dispatch
            benchmark_tracker.register_launched_request()
            input_queue.put(request_config)
            logger.info(f"Dispatched request {request_config.id}")
        else:
            time.sleep(0.01)


def process_results(
    output_queue: Queue,
    benchmark_tracker: BenchmarkTracker,
    generated_responses: List[Response],
    pbar: tqdm,
    stop_event: threading.Event,
) -> None:
    """Thread function to process results from the output queue."""
    # On stop, attempt to drain for a short grace period, then exit
    POLL_TIMEOUT_S = 0.1
    DRAIN_MAX_EMPTY_POLLS = 50  # ~5s
    consecutive_empty_polls_after_stop = 0
    while not stop_event.is_set() or (
        benchmark_tracker.error is None
        and benchmark_tracker.num_completed_requests < benchmark_tracker.num_requests
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
        benchmark_tracker.add_request_metrics(request_metrics)
        if generated_response is not None:
            generated_responses.append(generated_response)

        pbar.update(benchmark_tracker.num_completed_requests - pbar.n)


def run_main_loop(
    config: BenchmarkConfig,
    request_generator: BaseRequestGenerator,
    benchmark_tracker: BenchmarkTracker,
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
        client_config=config.client_config,
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
            benchmark_tracker,
            request_generator,
            stop_event,
        ),
    )

    processor_thread = Thread(
        target=process_results,
        args=(
            output_queue,
            benchmark_tracker,
            generated_responses,
            pbar,
            stop_event,
        ),
    )

    dispatcher_thread.start()
    processor_thread.start()

    # Monitor and wait for completion
    with benchmark_tracker:
        while not benchmark_tracker.should_stop():
            time.sleep(0.1)
        logger.info("Stopping the main loop.")
        if benchmark_tracker.stop_requested and benchmark_tracker.error is None:
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

    if benchmark_tracker.error is None:
        logger.info("Main loop completed.")
    else:
        raise benchmark_tracker.error


def run_benchmark(
    config: BenchmarkConfig,
):
    """Run the benchmark and return the in-memory metrics object.

    Args:
        config: The benchmark configuration.

    Returns:
        BenchmarkTracker containing the collected metrics (including the `MetricStore`).
    """

    prepare_benchmark_output_dir(config)

    logger.info(
        f"Benchmark output directory: {config.output_dir}"
    )

    setup_api_environment(
        api_key=config.api_key,
        api_url=config.api_url,
    )

    generated_responses: List[Response] = []

    assert (
        config.client_config.tokenizer is not None
    ), "Tokenizer is required."

    tokenizer = get_tokenizer(
        tokenizer_name=config.client_config.tokenizer,
        trust_remote_code=True,
    )

    request_generator = RequestGeneratorRegistry.get(
        config.request_generator_config.get_type(),
        config=config.request_generator_config,
        tokenizer=tokenizer,
        client_config=config.client_config,
    )

    max_requests = (
        request_generator.num_requests
        if config.request_generator_config.get_type()
        == RequestGeneratorType.LMEVAL
        else config.max_completed_requests
    )
    pbar = tqdm(total=max_requests)

    benchmark_tracker = BenchmarkTracker(
        max_requests=max_requests,
        timeout=config.timeout,
        metrics_config=config.metrics_config,
        output_dir=config.output_dir,
    )

    run_main_loop(
        config=config,
        request_generator=request_generator,
        benchmark_tracker=benchmark_tracker,
        generated_responses=generated_responses,
        pbar=pbar,
    )

    logger.info(
        f"Results for token benchmark for {config.client_config.model} queried with the {config.client_config.llm_api} api. {benchmark_tracker}"
    )

    benchmark_tracker.store_output()

    store_generated_texts(benchmark_tracker.output_dir, generated_responses)

    # lm-eval specific
    if (
        config.request_generator_config.get_type()
        == RequestGeneratorType.LMEVAL
    ):
        request_generator.get_responses(generated_responses)
        lmeval_results = request_generator.evaluate()
        logger.info(f"Results: {lmeval_results}")

        store_lmeval_results(benchmark_tracker.output_dir, lmeval_results)

    return benchmark_tracker


def main():
    configs = BenchmarkConfig.create_from_cli_args()

    logger.info(
        f"Running {len(configs)} benchmark configurations sequentially."
    )

    for i, config in enumerate(configs):
        logger.info(f"Running benchmark with config: {config}")
        logger.info(f"Starting benchmark [{i+1}/{len(configs)}]")

        random.seed(config.seed)
        run_benchmark(config=config)

        logger.info(f"Completed benchmark [{i+1}/{len(configs)}]")

    logger.info("All benchmarks completed.")


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    main()
