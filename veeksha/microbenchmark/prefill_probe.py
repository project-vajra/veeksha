import json
import os
from typing import Dict, List

import numpy as np

from veeksha.benchmark import run_benchmark
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import ClientConfig
from veeksha.config.generators.interval_generator.static_generator import (
    StaticRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.length_generator.fixed_generator import (
    FixedRequestLengthGeneratorConfig,
)
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.config.metrics import MetricsConfig
from veeksha.config.microbenchmark import MicrobenchmarkConfig, PrefillProbeConfig
from veeksha.logger import init_logger

# pyright: reportCallIssue=false, reportArgumentType=false
logger = init_logger(__name__)


class PrefillProbe:
    def __init__(self, microbenchmark_config: MicrobenchmarkConfig) -> None:
        self.micro_config = microbenchmark_config
        assert isinstance(
            self.micro_config.probe_config, PrefillProbeConfig
        ), "PrefillProbe requires PrefillProbeConfig"

        self.probe_config: PrefillProbeConfig = self.micro_config.probe_config
        self.prefill_times: Dict[int, List[float]] = {}

    def _build_benchmark_config(
        self, run_dir: str, prefill_tokens: int
    ) -> BenchmarkConfig:
        length_generator_config = FixedRequestLengthGeneratorConfig(
            prefill_tokens=prefill_tokens,
            decode_tokens=1,
        )

        request_generator_config = SyntheticRequestGeneratorConfig(
            interval_generator_config=StaticRequestIntervalGeneratorConfig(),
            length_generator_config=length_generator_config,
        )

        client_config = ClientConfig(
            model=self.micro_config.model,
            tokenizer=self.micro_config.tokenizer,
            num_clients=1,
            num_concurrent_requests_per_client=1,
        )

        metrics_config = MetricsConfig(
            output_dir=run_dir,
            should_write_metrics_to_wandb=False,
            wandb_project=self.micro_config.wandb_project,
            wandb_run_name=f"prefill_p{prefill_tokens}_{self.micro_config.model}",
        )

        return BenchmarkConfig(
            seed=self.micro_config.seed,
            timeout=self.micro_config.timeout,
            api_url=self.micro_config.api_url,
            api_key=self.micro_config.api_key,
            max_completed_requests=self.probe_config.num_requests_per_prefill_length,
            client_config=client_config,
            metrics_config=metrics_config,
            request_generator_config=request_generator_config,
        )

    def run(self):
        for prefill_value in self.probe_config.prefill_lengths:
            run_dir = os.path.join(self.micro_config.output_dir, str(prefill_value))

            if os.path.isdir(run_dir):
                logger.info(
                    f"Skipping profiling for prefill value = {prefill_value}..."
                )
                # Still need to load the data for training later
                json_file = os.path.join(run_dir, "request_level_metrics.json")
                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        ttft = data.get("ttft")
                        if ttft is None:
                            logger.warning(
                                "Key 'ttft' missing in %s; skipping.", json_file
                            )
                        else:
                            self.prefill_times[prefill_value] = ttft
                else:
                    logger.warning("Missing %s; skipping cached load.", json_file)
            else:
                config = self._build_benchmark_config(run_dir, prefill_value)

                os.makedirs(run_dir, exist_ok=True)
                logger.info(f"Running profiling for prefill value = {prefill_value}...")
                service_metrics = run_benchmark(config)
                logger.info(f"Run benchmark done")

                benchmark_output_dir = service_metrics.output_dir
                metrics_file = os.path.join(
                    benchmark_output_dir, "request_level_metrics.json"
                )
                assert os.path.exists(
                    metrics_file
                ), f"Could not find the result file for {benchmark_output_dir}"

                with open(metrics_file, "r") as f:
                    data = json.load(f)
                    ttft = data.get("ttft")
                    if ttft is None:
                        logger.warning(
                            "Key 'ttft' missing in %s; skipping.", metrics_file
                        )
                    else:
                        self.prefill_times[prefill_value] = ttft

            logger.info(f"Profiling for prefill value = {prefill_value} done")

        # log all the prefill times with their length
        prefill_stats = {}
        for prefill_value in self.probe_config.prefill_lengths:
            if (
                prefill_value not in self.prefill_times
                or not self.prefill_times[prefill_value]
            ):
                logger.warning(
                    f"No prefill times found for prefill_value {prefill_value}"
                )
                prefill_stats[str(prefill_value)] = {"count": 0}
                continue
            times = self.prefill_times[prefill_value]
            prefill_stats[str(prefill_value)] = {
                "count": len(times),
                "mean": float(np.mean(times)),
                "median": float(np.median(times)),
                "std": float(np.std(times)),
                "min": float(np.min(times)),
                "max": float(np.max(times)),
            }

        print(f"Prefill runtime stats: {prefill_stats}")

        prefill_stats_file = os.path.join(
            self.micro_config.output_dir, "prefill_stats.json"
        )
        with open(prefill_stats_file, "w") as f:
            json.dump(prefill_stats, f)
