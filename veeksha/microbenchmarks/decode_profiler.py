import json
import os
from dataclasses import replace
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import wandb

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.generators.interval_generator.static_generator import (
    StaticRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.length_generator.fixed_generator import (
    FixedRequestLengthGeneratorConfig,
)
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.logger import init_logger
from veeksha.run_benchmark import run_benchmark

# pyright: reportCallIssue=false, reportArgumentType=false
logger = init_logger(__name__)

# Number of concurrent requests per client for decode profiling
DECODE_NUM_CONCURRENT_REQUESTS_PER_CLIENT = 10
# Number of profiling iterations
DECODE_PROFILING_ITERATIONS = 10


class DecodeProfiler:
    def __init__(self, base_config: BenchmarkConfig, decode_config) -> None:
        self.base_config = base_config
        self.decode_config = decode_config
        self.context_lengths = decode_config.context_lengths
        self.batch_sizes = decode_config.batch_sizes
        self.decode_times: Dict[Tuple[int, int], List[int]] = {}
        self.base_dir = self.base_config.metrics_config.output_dir

    def extract_decode_times(self, output_dir: str, batch_size: int) -> list[int]:
        metrics_file = os.path.join(output_dir, "request_level_metrics.json")

        assert os.path.exists(metrics_file)

        # Read the metrics file
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)

        token_arrival_times = metrics_data["token_arrival_times"]
        tbt_values = metrics_data["tbt"]

        # Filter out empty sequences (requests with zero output tokens)
        non_empty_arrivals = [arr for arr in token_arrival_times if arr]

        if not non_empty_arrivals:
            raise ValueError(
                "No requests produced any output tokens. Cannot compute decode window."
            )

        if self.decode_config.engine_uses_mixed_batching:
            if len(non_empty_arrivals) < batch_size:
                raise ValueError(
                    f"Mixed batching requires at least {batch_size} requests with tokens, "
                    f"got {len(non_empty_arrivals)}. Decode profiling run insufficient."
                )

            first_tokens = sorted(arr[0] for arr in non_empty_arrivals)
            batch_saturation_first_token_time = first_tokens[batch_size - 1]
            latest_first_token_time = max(arr[0] for arr in non_empty_arrivals)

            assert (
                latest_first_token_time >= batch_saturation_first_token_time
            ), "No overlapping generation window found. The earliest request finished before the latest request started."

            window_start_time = batch_saturation_first_token_time
            window_end_time = latest_first_token_time
        else:
            # Find the latest first token time across all requests
            # (the time when all requests have started generating tokens)
            latest_first_token_time = max(arr[0] for arr in non_empty_arrivals)

            # Find the earliest last token time across all requests
            # (the time when the first request completes)
            earliest_last_token_time = min(arr[-1] for arr in non_empty_arrivals)

            assert (
                latest_first_token_time <= earliest_last_token_time
            ), "No overlapping generation window found. The earliest request finished before the latest request started."

            window_start_time = latest_first_token_time
            window_end_time = earliest_last_token_time

        logger.info(
            f"Analyzing decode batch runtime in window: [{window_start_time}, {window_end_time}]"
        )

        # Create filtered tbt values
        filtered_tbt_values = []

        # Filter both arrival times and tbt values to exclude empty sequences
        non_empty_pairs = [
            (arrival_times, tbts)
            for arrival_times, tbts in zip(token_arrival_times, tbt_values)
            if arrival_times and tbts
        ]

        for arrival_times, tbts in non_empty_pairs:
            # first arrival time corresponds to prefill latency
            assert len(arrival_times) - 1 == len(
                tbts
            ), f"{len(arrival_times) - 1} != {len(tbts)}"

            for arrival_time, tbt in zip(arrival_times[1:], tbts):
                if window_start_time <= arrival_time <= window_end_time:
                    filtered_tbt_values.append(tbt)
        return filtered_tbt_values

    def run(self):
        for batch_size, context_length in product(
            self.batch_sizes, self.context_lengths
        ):
            prefill_tokens = context_length
            num_iterations_per_prefill = np.ceil(
                context_length / self.decode_config.engine_chunk_size
            ).astype(int)

            # Calculate decode tokens dynamically like the original implementation
            decode_tokens = int(
                batch_size * num_iterations_per_prefill + DECODE_PROFILING_ITERATIONS
            )
            decode_tokens *= 2

            # Create new config for this run with proper replacements
            length_config = FixedRequestLengthGeneratorConfig(
                prefill_tokens=prefill_tokens, decode_tokens=decode_tokens
            )

            # Create a synthetic request generator config with fixed length
            request_gen_config = SyntheticRequestGeneratorConfig(
                interval_generator_config=StaticRequestIntervalGeneratorConfig(),
                length_generator_config=length_config,
            )

            num_requests = batch_size
            if self.decode_config.engine_uses_mixed_batching:
                num_requests += int(
                    np.ceil(DECODE_PROFILING_ITERATIONS / num_iterations_per_prefill)
                )

            num_clients = int(
                np.ceil(num_requests / DECODE_NUM_CONCURRENT_REQUESTS_PER_CLIENT)
            )

            # Create new client config with updated num_clients and force streaming API
            client_config = replace(
                self.base_config.client_config,
                num_clients=num_clients,
                llm_api="openai_chat",
                address_append_value="chat/completions",
            )

            run_dir = os.path.join(
                self.base_dir,
                f"{context_length}_{batch_size}",
            )

            # Create new metrics config with updated wandb run name and output dir
            metrics_config = replace(
                self.base_config.metrics_config,
                wandb_run_name=f"decode_cl{context_length}_bsz{batch_size}_{self.base_config.client_config.model}",
                output_dir=run_dir,
                should_write_metrics_to_wandb=False,
            )

            # Create a new config with all replacements
            config = replace(
                self.base_config,
                request_generator_config=request_gen_config,
                max_completed_requests=num_requests,
                client_config=client_config,
                metrics_config=metrics_config,
            )

            os.makedirs(run_dir, exist_ok=True)
            logger.info(
                f"Running profiling for decode context_length = {context_length} and batch_size = {batch_size}..."
            )
            service_metrics = run_benchmark(config)
            logger.info(f"Run benchmark done")
            if wandb.run:
                wandb.finish()

            benchmark_output_dir = service_metrics.output_dir
            self.decode_times[(context_length, batch_size)] = self.extract_decode_times(
                benchmark_output_dir, batch_size
            )

        # log all the decode times with their length
        tbt_stats = {}
        for (context_length, batch_size), times in self.decode_times.items():
            key = f"{context_length}_{batch_size}"
            if not times:
                logger.warning(f"No decode TBTs in analysis window for {key}")
                tbt_stats[key] = {"count": 0}
                continue
            tbt_stats[key] = {
                "count": len(times),
                "mean": float(np.mean(times)),
                "median": float(np.median(times)),
                "std": float(np.std(times)),
                "min": float(np.min(times)),
                "max": float(np.max(times)),
            }

        print(f"Decode runtime stats: {tbt_stats}")

        decode_stats_file = os.path.join(self.base_dir, "decode_stats.json")
        with open(decode_stats_file, "w") as f:
            json.dump(tbt_stats, f)
