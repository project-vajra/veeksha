import os
import threading
import time
from queue import Queue
from typing import Any, List

from tqdm import tqdm

from veeksha.logger import init_logger
from veeksha.new.client.registry import ClientRegistry
from veeksha.new.config.benchmark import BenchmarkConfig
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.thread_pool import ThreadPoolManager
from veeksha.new.core.tokenizer import (
    TokenizerProvider,
    build_hf_tokenizer_handle_from_model,
)
from veeksha.new.core.trace_recorder import TraceRecorder
from veeksha.new.evaluator.registry import EvaluatorRegistry
from veeksha.new.generator.session.registry import SessionGeneratorRegistry
from veeksha.new.health import HealthChecker
from veeksha.new.traffic.registry import TrafficSchedulerRegistry
from veeksha.new.types import ChannelModality
from veeksha.new.workers import CompletionWorker, DispatchWorker, PrefetchWorker
from veeksha.new.workers.client_runner import ClientRunnerManager
from veeksha.new.workers.prefetch import SharedSessionCounter

logger = init_logger(__name__)


def _init_pbar(max_sessions: int, benchmark_timeout: float):
    """Initialize progress bar based on benchmark mode.

    Returns:
        Tuple of (pbar, time_based_progress)
    """
    if max_sessions > 0:
        # Session-based progress
        pbar = tqdm(
            total=max_sessions,
            desc="Sessions",
            unit="sess",
            dynamic_ncols=True,
            bar_format="{desc}: {n}/{total} [{percentage:3.0f}%] | {rate_fmt} | Elapsed: {elapsed}",
        )
        return pbar, False
    else:
        # Time-based progress
        pbar = tqdm(
            total=int(benchmark_timeout),
            desc="Benchmark",
            unit="s",
            dynamic_ncols=True,
            bar_format="{desc}: {elapsed}/{total}s [{percentage:3.0f}%] | Sessions: {postfix}",
        )
        pbar.set_postfix_str("0")
        return pbar, True


def _update_pbar(
    pbar, time_based_progress: bool, elapsed: float, total_done: int, state: dict
):
    """Update progress bar with current state.

    Args:
        pbar: tqdm progress bar instance
        time_based_progress: True if using time-based mode
        elapsed: Elapsed time in seconds
        total_done: Total completed + errored sessions
        state: Dict with 'last_completed' and 'last_time_update' keys (mutated in place)
    """
    if time_based_progress:
        elapsed_int = int(elapsed)
        if elapsed_int > state["last_time_update"]:
            pbar.update(elapsed_int - state["last_time_update"])
            state["last_time_update"] = elapsed_int
        if total_done > state["last_completed"]:
            pbar.set_postfix_str(str(total_done))
            state["last_completed"] = total_done
    else:
        if total_done > state["last_completed"]:
            pbar.update(total_done - state["last_completed"])
            state["last_completed"] = total_done


def _monitor_for_completion(
    traffic_scheduler,
    evaluator,
    pool_manager,
    benchmark_start,
    benchmark_timeout,
    timeout_triggered,
    pre_timeout_request_ids,
    max_sessions,
):
    pbar, time_based_progress = _init_pbar(max_sessions, benchmark_timeout)
    pbar_state = {"last_completed": 0, "last_time_update": 0}

    try:
        while True:
            time.sleep(0.1)

            completed, errored, _ = evaluator.get_session_counts()
            total_done = completed + errored
            elapsed = time.monotonic() - benchmark_start

            _update_pbar(pbar, time_based_progress, elapsed, total_done, pbar_state)

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
                    evaluator.set_included_requests(pre_timeout_request_ids)
                    break
            elif all_prefetch_done and not traffic_scheduler.has_pending_work():
                logger.info("All sessions completed")
                break
    finally:
        pbar.close()


def run_main_loop(
    session_generator,
    traffic_scheduler,
    evaluator,
    client,
    runtime_config,
    trace_recorder=None,
    benchmark_start_time=None,
):
    """Run the main benchmark loop with all workers.

    Args:
        session_generator: Session generator to produce sessions
        traffic_scheduler: Traffic scheduler to manage dispatch timing
        evaluator: Evaluator to collect metrics
        client: LLM client for request execution
        runtime_config: Runtime configuration with thread counts
        trace_recorder: Optional trace recorder
        benchmark_start_time: Start time of the benchmark
    """
    logger.info("Starting main loop")
    if benchmark_start_time is None:
        benchmark_start_time = time.monotonic()

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
            "trace_recorder": trace_recorder,
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

    if trace_recorder:
        trace_recorder.start()

    # Start all threads
    client_runner.start()
    pool_manager.start_all()

    logger.info(
        f"Started {pool_manager.get_total_thread_count()} worker threads "
        f"and {client_runner.get_worker_count()} client workers"
    )

    benchmark_start = benchmark_start_time
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
            max_sessions=runtime_config.max_sessions,
        )

    except KeyboardInterrupt:
        logger.info("Interrupted, stopping")

    # Signal stop and join
    stop_event.set()
    pool_manager.join_pool("prefetch", timeout=1.0)
    pool_manager.join_pool("dispatch", timeout=1.0)

    if trace_recorder:
        trace_recorder.stop()

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

    # 2. Get traffic scheduler
    traffic_scheduler = TrafficSchedulerRegistry.get(
        benchmark_config.traffic_scheduler.get_type(),
        config=benchmark_config.traffic_scheduler,
        seed_manager=seed_manager,
    )

    benchmark_start_time = time.monotonic()

    # 3. Get evaluator
    evaluator = EvaluatorRegistry.get(
        benchmark_config.evaluator.get_type(),
        config=benchmark_config.evaluator,
        seed_manager=seed_manager,
        output_dir=f"{benchmark_config.output_dir}/metrics",
        benchmark_start_time=benchmark_start_time,
    )

    # 4. Get client
    client = ClientRegistry.get(
        benchmark_config.client.get_type(),
        config=benchmark_config.client,
        tokenizer_provider=tokenizer_provider,
    )

    # 5. Run the benchmark

    trace_recorder = None
    if benchmark_config.trace_recorder.enabled:
        # ensure output dirs exists for traces
        os.makedirs(f"{benchmark_config.output_dir}/traces", exist_ok=True)
        trace_recorder = TraceRecorder(
            f"{benchmark_config.output_dir}/traces",
            benchmark_start_time,
            benchmark_config.trace_recorder,
        )
        trace_recorder.start()

    os.makedirs(f"{benchmark_config.output_dir}/metrics", exist_ok=True)

    try:
        run_main_loop(
            session_generator=session_generator,
            traffic_scheduler=traffic_scheduler,
            evaluator=evaluator,
            client=client,
            runtime_config=benchmark_config.runtime,
            trace_recorder=trace_recorder,
            benchmark_start_time=benchmark_start_time,
        )
    finally:
        if trace_recorder:
            trace_recorder.stop()

    # 6. Finalize and save results
    result = evaluator.finalize()

    evaluator.save(f"{benchmark_config.output_dir}/metrics")

    # 7. Benchmark health checks
    if benchmark_config.trace_recorder.enabled:
        logger.info("Running health checks...")
        health_checker = HealthChecker(
            trace_file=f"{benchmark_config.output_dir}/traces/dispatch_trace.jsonl",
            metrics_file=f"{benchmark_config.output_dir}/metrics/request_level_metrics.jsonl",
            benchmark_config=benchmark_config,
        )
        health_checker.run_and_save(
            f"{benchmark_config.output_dir}/health_check_results.txt"
        )
    else:
        logger.info(
            "Health checks not run: trace recorder is disabled or content is not included."
        )

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
