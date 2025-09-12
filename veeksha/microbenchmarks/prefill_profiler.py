import glob
import json
import multiprocessing
import os
import platform
from dataclasses import replace
from typing import Dict, List

import numpy as np

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
from veeksha.constants.prefill_constants import *
from veeksha.logger import init_logger
from veeksha.run_benchmark import run_benchmark

# pyright: reportCallIssue=false, reportArgumentType=false
logger = init_logger(__name__)


# RMSE threshold for the prefill time predictor
PREFILL_RMSE_THRESHOLD = 0.05
# Number of Ray clients to use for prefill profiling
PREFILL_NUM_CLIENTS = 1
# Number of concurrent requests per client for prefill profiling
PREFILL_NUM_CONCURRENT_REQUESTS_PER_CLIENT = 1
# Number of completed requests to wait for before stopping the prefill profiling for a prompt length
PREFILL_MAX_NUM_COMPLETED_REQUESTS = 1
# Decode tokens when running the prefill profiler
PREFILL_PROFILER_DECODE_TOKENS = 1
# Model to train on the prefill values and prefill times
PREFILL_MODEL = "RandomForestRegressor"
# Random Forest Regressor parameters
PREFILL_RANDOM_FOREST_PARAMS = {
    "n_estimators": 10,
    "random_state": 0,
}


class PrefillProfiler:
    def __init__(
        self, base_config: BenchmarkConfig, prefill_lengths: List[int]
    ) -> None:
        self.base_config = base_config
        self.prefill_values = prefill_lengths
        self.prefill_times: Dict[int, List[float]] = {}

        # Create profiler-specific config using replace() to respect frozen design
        profiler_client_config = replace(
            base_config.client_config,
            num_clients=PREFILL_NUM_CLIENTS,
            num_concurrent_requests_per_client=PREFILL_NUM_CONCURRENT_REQUESTS_PER_CLIENT,
        )

        profiler_metrics_config = replace(
            base_config.metrics_config, should_write_metrics_to_wandb=False
        )

        self.config = replace(
            base_config,
            max_completed_requests=PREFILL_MAX_NUM_COMPLETED_REQUESTS,
            client_config=profiler_client_config,
            metrics_config=profiler_metrics_config,
            request_generator_config=SyntheticRequestGeneratorConfig(
                interval_generator_config=StaticRequestIntervalGeneratorConfig()
            ),
        )

        self.base_dir = self.base_config.metrics_config.output_dir

    def run(self):
        for prefill_value in self.prefill_values:
            # Create config for this specific prefill run using replace()
            length_generator_config = FixedRequestLengthGeneratorConfig(
                decode_tokens=PREFILL_PROFILER_DECODE_TOKENS,
                prefill_tokens=prefill_value,
            )

            request_generator_config = replace(
                self.config.request_generator_config,
                length_generator_config=length_generator_config,
            )

            run_config = replace(
                self.config, request_generator_config=request_generator_config
            )

            run_dir = os.path.join(
                self.base_dir,
                f"{self.config.client_config.model}_{prefill_value}",
            )

            if os.path.isdir(run_dir):
                logger.info(
                    f"Skipping profiling for prefill value = {prefill_value}..."
                )
                # Still need to load the data for training later
                json_file = os.path.join(run_dir, f"request_level_metrics.json")
                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        self.prefill_times[prefill_value] = data["ttft"]
            else:
                # Create final config with updated output dir and wandb name
                run_metrics_config = replace(
                    run_config.metrics_config,
                    wandb_run_name=f"prefill_p{prefill_value}_{self.config.client_config.model}",
                    output_dir=run_dir,
                )

                final_run_config = replace(
                    run_config, metrics_config=run_metrics_config
                )

                os.makedirs(run_dir, exist_ok=True)
                logger.info(f"Running profiling for prefill value = {prefill_value}...")
                service_metrics = run_benchmark(final_run_config)
                logger.info(f"Run benchmark done")

                json_file = os.path.join(run_dir, f"request_level_metrics.json")
                assert os.path.exists(
                    json_file
                ), f"Could not find the result file for {run_dir}"

                with open(json_file, "r") as f:
                    data = json.load(f)
                    self.prefill_times[prefill_value] = data["ttft"]

            logger.info(f"Profiling for prefill value = {prefill_value} done")

        # log all the prefill times with their length
        prefill_stats = {
            str(prefill_value): {
                "mean": float(np.mean(self.prefill_times[prefill_value])),
                "median": float(np.median(self.prefill_times[prefill_value])),
                "std": float(np.std(self.prefill_times[prefill_value])),
                "min": float(np.min(self.prefill_times[prefill_value])),
                "max": float(np.max(self.prefill_times[prefill_value])),
            }
            for prefill_value in self.prefill_values
        }

        print(f"Prefill runtime stats: {prefill_stats}")

        prefill_stats_file = os.path.join(self.base_dir, "prefill_stats.json")
        with open(prefill_stats_file, "w") as f:
            json.dump(prefill_stats, f)
