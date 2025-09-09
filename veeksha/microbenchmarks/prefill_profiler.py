import glob
import json
import multiprocessing
import os
import platform
from dataclasses import replace
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.generators.length_generator.fixed_generator import (
    FixedRequestLengthGeneratorConfig,
)
from veeksha.config.generators.interval_generator.static_generator import (
    StaticRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.constants.prefill_constants import *
from veeksha.logger import init_logger
from veeksha.run_benchmark import run_benchmark

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
    def __init__(self, base_config: BenchmarkConfig) -> None:
        self.base_config = base_config
        self.prefill_values = base_config.prefill_profiler_config.prefill_lengths
        self.prefill_times: Dict[int, List[float]] = {}
        self.model = RandomForestRegressor(
            n_estimators=PREFILL_RANDOM_FOREST_PARAMS["n_estimators"],
            random_state=PREFILL_RANDOM_FOREST_PARAMS["random_state"],
        )
        self.transformer = PolynomialFeatures(
            degree=PREFILL_POLYNOMIAL_DEGREE, include_bias=False
        )

        if PREFILL_MODEL != "RandomForestRegressor":
            raise NotImplementedError(f"Model {PREFILL_MODEL} is not implemented")

        # Create profiler-specific config using replace() to respect frozen design
        profiler_client_config = replace(
            base_config.client_config,
            num_clients=PREFILL_NUM_CLIENTS,
            num_concurrent_requests_per_client=PREFILL_NUM_CONCURRENT_REQUESTS_PER_CLIENT
        )
        
        profiler_metrics_config = replace(
            base_config.metrics_config,
            should_write_metrics_to_wandb=False
        )
        
        self.config = replace(
            base_config,
            max_completed_requests=PREFILL_MAX_NUM_COMPLETED_REQUESTS,
            client_config=profiler_client_config,
            metrics_config=profiler_metrics_config,
            request_generator_config=SyntheticRequestGeneratorConfig(
                interval_generator_config=StaticRequestIntervalGeneratorConfig()
            )
        )
        
        self.base_dir = self.base_config.metrics_config.output_dir

    def train_prefill_predictor_model(self):
        # Convert dictionary of lists to single arrays for training
        all_prefill_times = []
        all_prefill_values = []
        
        for prefill_value, times_list in self.prefill_times.items():
            # Use median time for each prefill value for training
            median_time = np.median(times_list)
            all_prefill_times.append(median_time)
            all_prefill_values.append(prefill_value)
        
        transformed_prefill_values = self.transformer.fit_transform(
            np.array(all_prefill_values).reshape(-1, 1)
        )

        self.model.fit(transformed_prefill_values, np.array(all_prefill_times))
        rmse = np.sqrt(
            np.mean(
                (
                    self.model.predict(transformed_prefill_values)
                    - np.array(all_prefill_times)
                )
                ** 2
            )
        )
        logger.info(
            f"Model fitted with prefill values and times with root mean squared error: {rmse}",
        )

        joblib.dump(
            self.model,
            os.path.join(self.base_dir, "prefill_predictor.pkl"),
        )

        # also plot the curve containing model's predictions and actual outputs, and dump it
        plt.figure(figsize=(10, 6))
        plt.plot(all_prefill_values, all_prefill_times, 'o', label="Actual")
        plt.plot(
            all_prefill_values,
            self.model.predict(transformed_prefill_values),
            'x',
            label="Predicted",
        )
        plt.xlabel("Prompt Length")
        plt.ylabel("Prefill Time")
        plt.title(self.config.client_config.model)
        plt.legend()
        plt.savefig(os.path.join(self.base_dir, "prefill_predictions.png"))

        # also do fine-grained plotting
        fine_grained_prefill_values = np.linspace(
            min(all_prefill_values), max(all_prefill_values), 1000
        )
        fine_grained_transformed_prefill_values = self.transformer.fit_transform(
            fine_grained_prefill_values.reshape(-1, 1)
        )
        fine_grained_prefill_times = self.model.predict(
            fine_grained_transformed_prefill_values
        )
        plt.plot(
            fine_grained_prefill_values,
            fine_grained_prefill_times,
            '-',
            label="Fine-grained Prediction",
        )
        plt.xlabel("Prompt Length")
        plt.ylabel("Prefill Time")
        plt.title(self.config.client_config.model)
        plt.legend()
        plt.savefig(
            os.path.join(
                self.base_dir,
                "fine_grained_prefill_predictions.png",
            )
        )

        plt.close()

        if (
            self.config.metrics_config.wandb_project
            and self.config.metrics_config.should_write_metrics
        ):
            wandb.init(
                project=self.config.metrics_config.wandb_project,
                group=self.config.metrics_config.wandb_group,
                name=f"prefill_profiler_{self.config.client_config.model}",
            )
            data = {
                "prefill_lengths": all_prefill_values,
                "prefill_times": all_prefill_times,
            }
            wandb.log(
                {
                    "prefill_times_vs_length": wandb.plot.line(
                        table=wandb.Table(data=pd.DataFrame(data)),
                        x="prefill_lengths",
                        y="prefill_times",
                        title="Prefill Times vs Prefill Lengths",
                    )
                },
                step=0,
            )
            data = {
                "prefill_lengths": fine_grained_prefill_values,
                "predicted_prefill_times": fine_grained_prefill_times,
            }
            wandb.log(
                {
                    "predicted_prefill_times_vs_length": wandb.plot.line(
                        table=wandb.Table(data=pd.DataFrame(data)),
                        x="prefill_lengths",
                        y="predicted_prefill_times",
                        title="Predicted Prefill Times vs Prefill Lengths",
                    )
                },
                step=0,
            )

        if self.config.prefill_profiler_config.cache_predictions:
            predictions = {}

            x = np.arange(
                self.config.prefill_profiler_config.max_prefill_tokens_to_predict + 1
            )
            x = x.reshape(-1, 1)
            x_poly = self.transformer.fit_transform(x)
            y = self.model.predict(x_poly)
            for i in range(len(x)):
                predictions[int(x[i][0])] = float(y[i])

            joblib.dump(
                predictions,
                os.path.join(self.base_dir, "prefill_predictions.pkl"),
            )

    def run(self):
        for prefill_value in self.prefill_values:
            # Create config for this specific prefill run using replace()
            length_generator_config = FixedRequestLengthGeneratorConfig(
                decode_tokens=PREFILL_PROFILER_DECODE_TOKENS,
                prefill_tokens=prefill_value,
            )
            
            request_generator_config = replace(
                self.config.request_generator_config,
                length_generator_config=length_generator_config
            )
            
            run_config = replace(
                self.config,
                request_generator_config=request_generator_config
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
                    output_dir=run_dir
                )
                
                final_run_config = replace(
                    run_config,
                    metrics_config=run_metrics_config
                )
                
                os.makedirs(run_dir, exist_ok=True)
                logger.info(f"Running profiling for prefill value = {prefill_value}...")
                service_metrics = run_benchmark(final_run_config)
                logger.info(f"Run benchmark done")
                if wandb.run:
                    wandb.finish()

                json_file = os.path.join(run_dir, f"request_level_metrics.json")
                assert os.path.exists(json_file), f"Could not find the result file for {run_dir}"

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
            } for prefill_value in self.prefill_values}

        print(f"Prefill runtime stats: {prefill_stats}")

        prefill_stats_file = os.path.join(self.base_dir, "prefill_stats.json")
        with open(prefill_stats_file, "w") as f:
            json.dump(prefill_stats, f)

        if self.config.prefill_profiler_config.should_train_predictor:
            self.train_prefill_predictor_model()


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    configs = BenchmarkConfig.create_from_cli_args()
    config = configs[0] if isinstance(configs, list) else configs
    prefill_profiler = PrefillProfiler(config)
    prefill_profiler.run()