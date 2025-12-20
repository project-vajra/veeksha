import time
from typing import Any, List

from veeksha.logger import init_logger
from veeksha.new.client.registry import ClientRegistry
from veeksha.new.config.benchmark import BenchmarkConfig
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.tokenizer import (
    TokenizerProvider,
    build_hf_tokenizer_handle_from_model,
)
from veeksha.new.evaluator.registry import EvaluatorRegistry
from veeksha.new.generator.session.registry import SessionGeneratorRegistry
from veeksha.new.traffic.registry import TrafficSchedulerRegistry
from veeksha.new.types import ChannelModality

logger = init_logger(__name__)


def _monitor_for_completion(
    traffic_scheduler,
    evaluator,
    pool_manager,
    benchmark_start,
    benchmark_timeout,
    timeout_triggered,
    pre_timeout_request_ids,
):
    while True:
        time.sleep(0.1)

        elapsed = time.monotonic() - benchmark_start

        # check timeout if not already triggered
        if (
            not timeout_triggered
            and benchmark_timeout > 0
            and elapsed >= benchmark_timeout
        ):
            timeout_triggered = True
            pre_timeout_request_ids = evaluator.get_registered_request_ids()
            in_flight = traffic_scheduler.get_in_flight_request_ids()
            # only care about pre-timeout requests that are still in-flight
            pending = pre_timeout_request_ids & in_flight
            logger.info(
                f"Benchmark timeout after {elapsed:.1f}s. "
                f"Captured {len(pre_timeout_request_ids)} registered requests, "
                f"{len(pending)} still in-flight."
            )

        # check if all prefetch workers have finished
        prefetch_threads = pool_manager.thread_pools.get("prefetch", [])
        all_prefetch_done = all(not t.is_alive() for t in prefetch_threads)

        if timeout_triggered:
            # wait for pre-timeout requests to complete
            current_in_flight = traffic_scheduler.get_in_flight_request_ids()
            remaining = pre_timeout_request_ids & current_in_flight
            if not remaining:
                logger.info("All pre-timeout requests completed")
                # notify evaluator to only include pre-timeout requests
                evaluator.set_included_requests(pre_timeout_request_ids)
                break
        elif all_prefetch_done and not traffic_scheduler.has_pending_work():
            # normal completion: all sessions generated and all work done
            logger.info("All sessions completed")
            break


def run_main_loop(
    session_generator,
    traffic_scheduler,
    evaluator,
    client,
    runtime_config,
):
    """Run the main benchmark loop with all workers.

    Args:
        session_generator: Session generator to produce sessions
        traffic_scheduler: Traffic scheduler to manage dispatch timing
        evaluator: Evaluator to collect metrics
        client: LLM client for request execution
        runtime_config: Runtime configuration with thread counts
    """
    import threading
    import time
    from queue import Queue

    from veeksha.new.core.thread_pool import ThreadPoolManager
    from veeksha.new.workers import CompletionWorker, DispatchWorker, PrefetchWorker
    from veeksha.new.workers.client_runner import ClientRunnerManager
    from veeksha.new.workers.prefetch import SharedSessionCounter

    logger.info("Starting main loop")

    # Create queues
    client_queues = [Queue() for _ in range(runtime_config.num_client_threads)]
    output_queue = Queue()
    stop_event = threading.Event()
    generator_lock = threading.Lock()

    session_counter = SharedSessionCounter(max_sessions=runtime_config.max_sessions)

    client_runner = ClientRunnerManager(
        client=client,
        input_queues=client_queues,
        output_queue=output_queue,
        stop_event=stop_event,
    )

    # Create thread pool manager
    pool_manager = ThreadPoolManager(stop_event=stop_event)

    # Create worker pools
    pool_manager.create_pool(
        name="prefetch",
        worker_class=PrefetchWorker,
        worker_kwargs={
            "traffic_scheduler": traffic_scheduler,
            "session_generator": session_generator,
            "generator_lock": generator_lock,
            "session_counter": session_counter,
        },
        pool_size=1,  # generation is sequential because from trace needs ordering
    )

    pool_manager.create_pool(
        name="dispatch",
        worker_class=DispatchWorker,
        worker_kwargs={
            "traffic_scheduler": traffic_scheduler,
            "client_queues": client_queues,
            "evaluator": evaluator,
        },
        pool_size=runtime_config.num_dispatcher_threads,
    )

    pool_manager.create_pool(
        name="completion",
        worker_class=CompletionWorker,
        worker_kwargs={
            "output_queue": output_queue,
            "traffic_scheduler": traffic_scheduler,
            "evaluator": evaluator,
        },
        pool_size=runtime_config.num_completion_threads,
    )

    # Start all threads
    client_runner.start()
    pool_manager.start_all()

    logger.info(
        f"Started {pool_manager.get_total_thread_count()} worker threads "
        f"and {client_runner.get_worker_count()} client workers"
    )

    benchmark_start = time.monotonic()
    benchmark_timeout = runtime_config.benchmark_timeout
    timeout_triggered = False
    pre_timeout_request_ids: set = set()

    try:
        _monitor_for_completion(
            traffic_scheduler,
            evaluator,
            pool_manager,
            benchmark_start,
            benchmark_timeout,
            timeout_triggered,
            pre_timeout_request_ids,
        )

    except KeyboardInterrupt:
        logger.info("Interrupted, stopping")

    # Signal stop and join
    stop_event.set()
    pool_manager.join_pool("prefetch", timeout=1.0)
    pool_manager.join_pool("dispatch", timeout=1.0)

    # Stop client runner
    client_runner.stop()
    client_runner.wait()

    # Send sentinels to completion workers and join
    for _ in range(runtime_config.num_completion_threads):
        output_queue.put(None)
    pool_manager.join_pool("completion", timeout=1.0)

    logger.info("Main loop completed")


def run_benchmark(
    benchmark_config: BenchmarkConfig,
):
    """Run the benchmark and return evaluation results.

    Args:
        benchmark_config: The benchmark configuration.

    Returns:
        EvaluationResult from the evaluator.
    """
    logger.info("Running benchmark with config:\n%s", benchmark_config)

    seed_manager = SeedManager(benchmark_config.seed)

    # 1. Get session generator
    logger.info(
        f"Session generator type: {benchmark_config.session_generator.get_type()}"
    )
    tokenizer_provider = TokenizerProvider(
        {
            ChannelModality.TEXT: build_hf_tokenizer_handle_from_model(
                benchmark_config.client.model
            )
        }
    )
    append_min_tokens_instruction = False
    if (
        hasattr(benchmark_config.client, "use_min_tokens_prompt_fallback")
        and benchmark_config.client.use_min_tokens_prompt_fallback  # type: ignore
    ):
        append_min_tokens_instruction = True
        logger.info(
            "Min tokens prompt fallback enabled in config. "
            "Will append instructions to prompts for minimum token control."
        )

    session_generator_kwargs = {
        "config": benchmark_config.session_generator,
        "seed_manager": seed_manager,
        "tokenizer_provider": tokenizer_provider,
    }

    if (
        benchmark_config.session_generator.get_type()
        == SessionGeneratorRegistry.get_key_from_str("synthetic")
    ):
        session_generator_kwargs["append_min_tokens_instruction"] = (
            append_min_tokens_instruction
        )

    session_generator = SessionGeneratorRegistry.get(
        benchmark_config.session_generator.get_type(),
        **session_generator_kwargs,
    )
    logger.info(f"Session generator: {session_generator}")

    # 2. Get traffic scheduler
    traffic_scheduler = TrafficSchedulerRegistry.get(
        benchmark_config.traffic_scheduler.get_type(),
        config=benchmark_config.traffic_scheduler,
        seed_manager=seed_manager,
    )
    logger.info(f"Traffic scheduler: {traffic_scheduler}")

    # 3. Get evaluator
    evaluator = EvaluatorRegistry.get(
        benchmark_config.evaluator.get_type(),
        config=benchmark_config.evaluator,
        seed_manager=seed_manager,
    )
    logger.info(f"Evaluator: {evaluator}")

    # 4. Get client
    client = ClientRegistry.get(
        benchmark_config.client.get_type(),
        config=benchmark_config.client,
        tokenizer_provider=tokenizer_provider,
    )
    logger.info(f"Client: {client}")

    # 5. Run the benchmark
    run_main_loop(
        session_generator=session_generator,
        traffic_scheduler=traffic_scheduler,
        evaluator=evaluator,
        client=client,
        runtime_config=benchmark_config.runtime,
    )

    # 6. Finalize and save results
    result = evaluator.finalize()
    evaluator.save(benchmark_config.evaluator.output_dir)

    logger.info("Benchmark completed")
    return result


def run_benchmarks(benchmark_configs: List[BenchmarkConfig]) -> List[Any]:
    """Run benchmarks sequentially.

    Args:
        benchmark_configs: List of configurations for the benchmarks

    Returns:
        List of EvaluationResult objects from each benchmark.
    """
    results: List[Any] = []

    if len(benchmark_configs) > 1:
        logger.info(
            f"Running {len(benchmark_configs)} benchmark configurations sequentially."
        )

    for i, benchmark_config in enumerate(benchmark_configs):
        logger.info(f"Running benchmark {i+1}/{len(benchmark_configs)}")
        result = run_benchmark(benchmark_config=benchmark_config)
        results.append(result)
        logger.info(f"Completed benchmark {i+1}/{len(benchmark_configs)}")

    logger.info("All benchmarks completed.")
    return results


if __name__ == "__main__":
    pass
