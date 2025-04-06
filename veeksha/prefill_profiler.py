import glob
import json
import multiprocessing
import os
import platform

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

from veeksha.config.config import (
    BenchmarkConfig,
    FixedRequestLengthGeneratorConfig,
    StaticRequestIntervalGeneratorConfig,
    SyntheticRequestGeneratorConfig
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
PREFILL_MAX_NUM_COMPLETED_REQUESTS = 20
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
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.prefill_values = self.config.prefill_profiler_config.prefill_lengths
        self.prefill_times: Dict[int, List[int]] = {}
        self.model = RandomForestRegressor(
            n_estimators=PREFILL_RANDOM_FOREST_PARAMS["n_estimators"],
            random_state=PREFILL_RANDOM_FOREST_PARAMS["random_state"],
        )
        self.transformer = PolynomialFeatures(
            degree=PREFILL_POLYNOMIAL_DEGREE, include_bias=False
        )

        if PREFILL_MODEL != "RandomForestRegressor":
            raise NotImplementedError(f"Model {PREFILL_MODEL} is not implemented")

        # update the config with some fixed constants
        self.config.request_generator_config = SyntheticRequestGeneratorConfig()
        self.config.request_interval_generator_config = StaticRequestIntervalGeneratorConfig()
        self.config.metrics_config.should_write_metrics = False
        self.config.client_config.num_clients = PREFILL_NUM_CLIENTS
        self.config.client_config.num_concurrent_requests_per_client = (
            PREFILL_NUM_CONCURRENT_REQUESTS_PER_CLIENT
        )
        self.config.max_completed_requests = PREFILL_MAX_NUM_COMPLETED_REQUESTS
        self.base_dir = self.config.metrics_config.output_dir

    def train_prefill_predictor_model(self):
        # TODO(Amey/Anmol): Prefill times is now a dictionary of lists
        transformed_prefill_values = self.transformer.fit_transform(
            np.array(self.prefill_values).reshape(-1, 1)
        )

        self.model.fit(transformed_prefill_values, np.array(self.prefill_times))
        rmse = np.sqrt(
            np.mean(
                (
                    self.model.predict(transformed_prefill_values)
                    - np.array(self.prefill_times)
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
        plt.plot(self.prefill_values, self.prefill_times, label="Actual")
        plt.plot(
            self.prefill_values,
            self.model.predict(transformed_prefill_values),
            label="Predicted",
        )
        plt.xlabel("Prompt Length")
        plt.ylabel("Prefill Time")
        plt.title(self.config.client_config.model)
        plt.legend()
        plt.savefig(os.path.join(self.base_dir, "prefill_predictions.png"))

        # also do fine-grained plotting
        fine_grained_prefill_values = np.linspace(
            min(self.prefill_values), max(self.prefill_values), 1000
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
                name=f"prefill_profiler_{self.config.client_config.model}_{self.config.timestamp}",
            )
            data = {
                "prefill_lengths": self.prefill_values,
                "prefill_times": self.prefill_times,
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

        # assert rmse < PREFILL_RMSE_THRESHOLD, "Model's RMSE is too high, consider changing the model or the data"

        if self.config.prefill_profiler_config.cache_predictions:
            predictions = {}

            x = np.arange(
                self.config.prefill_profiler_config.max_prefill_tokens_to_predict + 1
            )
            x = x.reshape(-1, 1)
            x_poly = self.transformer.fit_transform(x)
            y = self.model.predict(x_poly)
            for i in range(len(x)):
                predictions[int(x[i][0])] = y[i]

            joblib.dump(
                predictions,
                os.path.join(self.base_dir, "prefill_predictions.pkl"),
            )


    def run(self):
        for prefill_value in self.prefill_values:
            self.config.request_generator_config.length_generator_config = FixedRequestLengthGeneratorConfig(
                decode_tokens=1,
                prefill_tokens=prefill_value,
                max_tokens=prefill_value + 1,
            )

            run_dir = os.path.join(
                self.base_dir,
                f"{self.config.client_config.model}_{prefill_value}",
            )

            self.config.metrics_config.wandb_run_name = (
                f"prefill_p{prefill_value}_{self.config.client_config.model}"
            )
            self.config.metrics_config.output_dir = run_dir
            os.makedirs(run_dir, exist_ok=True)
            logger.info(f"Running profiling for prefill value = {prefill_value}...")
            run_benchmark(self.config)
            logger.info(f"Run benchmark done")
            if wandb.run:
                wandb.finish()

            json_file = os.path.join(run_dir, f"request_level_metrics.json")

            assert os.path.exists(json_file), f"Could not find the result file for {run_dir}"

            with open(json_file, "r") as f:
                data = json.load(f)
                self.prefill_times[prefill_value] = data["ttft"]

        # log all the prefill times with their length
        tbt_stats = {
            prefill_value: {
                "mean": np.mean(self.prefill_times[prefill_value]),
                "median": np.median(self.prefill_times[prefill_value]),
                "std": np.std(self.prefill_times[prefill_value]),
                "min": np.min(self.prefill_times[prefill_value]),
                "max": np.max(self.prefill_times[prefill_value]),
            } for prefill_value in self.prefill_values}

        print(f"Prefill runtime stats: {tbt_stats}")

        prefill_stats_file = os.path.join(self.base_dir, "prefill_stats.json")
        with open(prefill_stats_file, "w") as f:
            json.dump(tbt_stats, f)

        if self.config.prefill_profiler_config.should_train_predictor:
            self.train_prefill_predictor_model()


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    config: BenchmarkConfig = BenchmarkConfig.create_from_cli_args()
    prefill_profiler = PrefillProfiler(config)
    prefill_profiler.run()

