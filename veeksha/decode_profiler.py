import json
import multiprocessing
import os
import platform
from itertools import product

import numpy as np
import pandas as pd
import wandb

from veeksha.config.config import (
    BenchmarkConfig,
    FixedRequestLengthGeneratorConfig,
    StaticRequestIntervalGeneratorConfig,
    SyntheticRequestGeneratorConfig
)
from veeksha.logger import init_logger
from veeksha.run_benchmark import run_benchmark

logger = init_logger(__name__)

# Number of concurrent requests per client for decode profiling
DECODE_NUM_CONCURRENT_REQUESTS_PER_CLIENT = 10
# Number of profiling iterations
DECODE_PROFILING_ITERATIONS = 10


class DecodeProfiler:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.context_lengths = self.config.decode_profiler_config.context_lengths
        self.batch_sizes = self.config.decode_profiler_config.batch_sizes
        self.decode_times: Dict[Tuple[int, int], List[int]] = {}

        # update the config with some fixed constants
        self.config.request_generator_config = SyntheticRequestGeneratorConfig()
        self.config.request_interval_generator_config = StaticRequestIntervalGeneratorConfig()
        self.config.metrics_config.should_write_metrics = False
        self.config.client_config.num_concurrent_requests_per_client = (
            DECODE_NUM_CONCURRENT_REQUESTS_PER_CLIENT
        )
        self.base_dir = self.config.metrics_config.output_dir

    def extract_decode_times(self, output_dir: str) -> list[int]:
        metrics_file = os.path.join(output_dir, "request_level_metrics.json")

        assert os.path.exists(metrics_file)
        
        # Read the metrics file
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)

        token_arrival_times = metrics_data["token_arrival_times"]
        tbt_values = metrics_data["tbt"]
        
        # Find the latest first token time across all requests 
        # (the time when all requests have started generating tokens)
        latest_first_token_time = max(arr[0] for arr in token_arrival_times)
        
        # Find the earliest last token time across all requests
        # (the time when the first request completes)
        earliest_last_token_time = min(arr[-1] for arr in token_arrival_times)
        
        assert latest_first_token_time <= earliest_last_token_time, "No overlapping generation window found. The earliest request finished before the latest request started."

        logger.info(f"Analyzing decode batch runtime in window: [{latest_first_token_time}, {earliest_last_token_time}]")

        # Create filtered tbt values
        filtered_tbt_values = []
        
        for arrival_times, tbts in zip(token_arrival_times, tbt_values):
            assert arrival_times 
            assert tbts
            # first arrival time corrosponds to prefill latency
            assert len(arrival_times) - 1 == len(tbts), f"{len(arrival_times) - 2} != {len(tbts)}"

            for arrival_time, tbt in zip(arrival_times[1:], tbts):
                if latest_first_token_time <= arrival_time <= earliest_last_token_time:
                    filtered_tbt_values.append(tbt)
        return filtered_tbt_values

    def run(self):
        for batch_size, context_length in product(self.batch_sizes, self.context_lengths):
            prefill_tokens = context_length
            num_iterations_per_prefill = np.ceil(
                context_length / self.config.decode_profiler_config.engine_chunk_size
            ).astype(int)

            decode_tokens = batch_size * num_iterations_per_prefill + DECODE_PROFILING_ITERATIONS
            decode_tokens *= 2

            self.config.request_generator_config.length_generator_config = FixedRequestLengthGeneratorConfig(
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                max_tokens=prefill_tokens + decode_tokens
            )
            self.config.max_completed_requests = batch_size
            self.config.client_config.num_clients = np.ceil(batch_size / DECODE_NUM_CONCURRENT_REQUESTS_PER_CLIENT).astype(int)

            run_dir = os.path.join(
                self.base_dir,
                f"{self.config.client_config.model}_{context_length}_{batch_size}",
            )

            self.config.metrics_config.wandb_run_name = (
                f"decode_cl{context_length}_bsz{batch_size}_{self.config.client_config.model}"
            )
            self.config.metrics_config.output_dir = run_dir
            os.makedirs(run_dir, exist_ok=True)
            logger.info(f"Running profiling for decode context_length = {context_length} and batch_size = {batch_size}...")
            run_benchmark(self.config)
            logger.info(f"Run benchmark done")
            if wandb.run:
                wandb.finish()

            self.decode_times[(context_length, batch_size)] = self.extract_decode_times(run_dir)

        # log all the decode times with their length
        tbt_stats = {
            f"{context_length}_{batch_size}": {
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
            } for (context_length, batch_size), times in self.decode_times.items()}

        print(f"Decode runtime stats: {tbt_stats}")

        decode_stats_file = os.path.join(self.base_dir, "decode_stats.json")
        with open(decode_stats_file, "w") as f:
            json.dump(tbt_stats, f)


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    config: BenchmarkConfig = BenchmarkConfig.create_from_cli_args()
    decode_profiler = DecodeProfiler(config)
    decode_profiler.run()

