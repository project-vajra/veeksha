import glob
import hashlib
import json
import os
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import wandb

from veeksha.capacity_search.benchmark_wrapper import run
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.config.utils import create_class_from_dict, dataclass_to_dict
from veeksha.logger import init_logger

logger = init_logger(__name__)

# Increase upper bound of QPS by this scale during binary search
QPS_INCREASE_SCALE = 2
# Threshold to increase the upper bound of QPS during binary search
VICINITY_THRESHOLD = 0.8


class CapacitySearch:
    def __init__(
        self,
        capacity_search_config: CapacitySearchConfig,
        benchmark_config_params: Dict,
    ) -> None:
        self.capacity_search_config = capacity_search_config
        self.benchmark_config_params = benchmark_config_params
        # we need this to get default values, with which we set the output_dir
        self.default_benchmark_config = create_class_from_dict(
            BenchmarkConfig, benchmark_config_params
        )

        self.stop_event = threading.Event()

        self.full_config = {
            "capacity_search_config": dataclass_to_dict(self.capacity_search_config),
            "benchmark_config": dataclass_to_dict(self.default_benchmark_config),
        }

        model_name = self.default_benchmark_config.client_config.model.split("/")[-1]
        config_hash = hashlib.md5(str(self.full_config).encode()).hexdigest()[:8]

        # TODO add params
        # for example
        # output_dir=os.path.join(
        #         self.args.output_dir,
        #         str(self.job_config.server_config.openai_server_engine),
        #         self.job_config.model_config.name,
        #         # f"ttft_slack_{self.args.ttft_slack_slo}_tbt_{self.args.tbt_slo}",
        #         str(self.job_config.request_generator_config.trace_file_name),
        #         f"{hash_key}_q{qps}",
        #     )
        self.job_output_dir = os.path.join(
            self.capacity_search_config.output_dir, f"{model_name}_{config_hash}"
        )
        os.makedirs(self.job_output_dir, exist_ok=True)

        with open(os.path.join(self.job_output_dir, "config.json"), "w") as f:
            json.dump(self.full_config, f, indent=4)

        if (
            (self.capacity_search_config.slo_type == "deadline")
            and self.capacity_search_config.dynamic_ttft_slo
            and self.default_benchmark_config.prefill_profiler_config.use_predictions_for_ttft
        ):
            assert (
                self.default_benchmark_config.prefill_profiler_config.predictor_dir
                is not None
            ), "Deadline SLO needs predictor directory"

    def _run_capacity_search_benchmark(
        self, qps: float
    ) -> Tuple[
        bool, Optional[float], Optional[float], Optional[float], Optional[float], str
    ]:
        qps_run_dir = os.path.join(self.job_output_dir, str(qps))

        # each run has a different qps, which must be reflected both in the wandb run name and the metrics output directory
        if self.benchmark_config_params.get("metrics_config"):
            self.benchmark_config_params["metrics_config"][
                "wandb_run_name"
            ] = f"qps_{qps}_model_{self.default_benchmark_config.client_config.model}"
            self.benchmark_config_params["metrics_config"]["output_dir"] = qps_run_dir
        else:
            self.benchmark_config_params["metrics_config"] = {
                "wandb_run_name": f"qps_{qps}_model_{self.default_benchmark_config.client_config.model}",
                "output_dir": qps_run_dir,
            }

        benchmark_config = create_class_from_dict(
            BenchmarkConfig, self.benchmark_config_params
        )

        cached_request_level_metrics_file = self._get_request_level_metrics(qps_run_dir)

        if cached_request_level_metrics_file is not None:
            logger.info(f"Cached results found for {qps}")
            return self._is_under_sla(cached_request_level_metrics_file, qps)

        run(benchmark_config)

        request_level_metrics_file = self._get_request_level_metrics(qps_run_dir)

        assert (
            request_level_metrics_file is not None
        ), f"Service-level metrics file not found for QPS: {qps}"

        return self._is_under_sla(request_level_metrics_file, qps)

    def _get_result_file(self, run_dir: str, metric_name: str) -> Optional[str]:
        files = glob.glob(os.path.join(run_dir, f"{metric_name}.csv"))
        if len(files) == 0:
            return None

        return files[0]

    def _get_request_level_metrics(self, run_dir: str) -> Optional[str]:
        files = glob.glob(os.path.join(run_dir, f"request_level_metrics.json"))
        if len(files) == 0:
            return None

        return files[0]

    def _get_service_level_metrics(self, run_dir: str) -> Optional[str]:
        files = glob.glob(os.path.join(run_dir, f"service_level_metrics.json"))
        if len(files) == 0:
            return None

        return files[0]

    def _use_deadline_based_slo(
        self, request_level_metrics_file: str
    ) -> Tuple[bool, float]:
        with open(request_level_metrics_file, "r") as f:
            request_level_metrics = json.load(f)

        deadline_miss_rate_array = request_level_metrics["deadline_miss_rate"]

        # Calculate percentile values of deadline miss rate
        deadline_miss_rate = np.quantile(
            deadline_miss_rate_array,
            self.capacity_search_config.deadline_miss_rate_percentile,
        )

        is_under_sla = (
            deadline_miss_rate <= self.capacity_search_config.deadline_miss_rate_slo
        )

        return is_under_sla, deadline_miss_rate

    def _use_tbt_and_ttft_slo(
        self,
        request_level_metrics_file: str,
    ) -> Tuple[bool, float, float]:
        with open(request_level_metrics_file, "r") as f:
            request_level_metrics = json.load(f)

        # Get TTFT, TBT request level
        ttft_array = request_level_metrics["ttft"]
        tbt_array = request_level_metrics["tbt"]

        # Merge TBT arrays of each request to make it service level
        combined_tbt_array = []
        for i in range(len(tbt_array)):
            combined_tbt_array.extend(tbt_array[i])

        # Calculate percentile values of TBT, TTFT
        tbt = np.quantile(
            combined_tbt_array, self.capacity_search_config.tbt_percentile
        )
        ttft = np.quantile(ttft_array, self.capacity_search_config.ttft_percentile)

        is_under_sla = (
            tbt <= self.capacity_search_config.tbt_slo
            and ttft <= self.capacity_search_config.ttft_slo
        )

        return is_under_sla, tbt, ttft

    def _use_ttft_and_tpot_slo(
        self,
        request_level_metrics_file: str,
    ) -> Tuple[bool, float, float]:
        with open(request_level_metrics_file, "r") as f:
            request_level_metrics = json.load(f)

        # Get TTFT, TPOT at request level
        ttft_array = request_level_metrics["ttft"]
        tpot_array = request_level_metrics["tpot"]

        # Calculate percentile values of TTFT, TPOT
        ttft = np.quantile(ttft_array, self.capacity_search_config.ttft_percentile)
        tpot = np.quantile(tpot_array, self.capacity_search_config.tpot_percentile)

        is_under_sla = (
            ttft <= self.capacity_search_config.ttft_slo
            and tpot <= self.capacity_search_config.tpot_slo
        )

        return is_under_sla, ttft, tpot

    def _is_under_sla(
        self,
        request_level_metrics_file: str,
        qps: float,
    ) -> Tuple[
        bool, Optional[float], Optional[float], Optional[float], Optional[float], str
    ]:
        is_under_sla = False
        tbt = None
        ttft = None
        tpot = None
        deadline_miss_rate = None

        if self.capacity_search_config.slo_type == "deadline":
            is_under_sla, deadline_miss_rate = self._use_deadline_based_slo(
                request_level_metrics_file
            )
        elif self.capacity_search_config.slo_type == "tbt_ttft":
            is_under_sla, tbt, ttft = self._use_tbt_and_ttft_slo(
                request_level_metrics_file
            )
        elif self.capacity_search_config.slo_type == "ttft_tpot":
            is_under_sla, ttft, tpot = self._use_ttft_and_tpot_slo(
                request_level_metrics_file
            )
        else:
            raise ValueError(
                f"Invalid SLO type: {self.capacity_search_config.slo_type}"
            )

        logger.info(
            f"QPS: {qps}"
            f" - TBT P{self.capacity_search_config.tbt_percentile * 100} Tokens: {tbt}"
            f" - TTFT P{self.capacity_search_config.ttft_percentile * 100} Tokens: {ttft}"
            f" - TPOT P{self.capacity_search_config.tpot_percentile * 100} Requests: {tpot}"
            f" - Deadline Miss Rate P{self.capacity_search_config.deadline_miss_rate_percentile * 100} Requests: {deadline_miss_rate}",
        )
        return (
            is_under_sla,
            tbt,
            ttft,
            tpot,
            deadline_miss_rate,
            str(qps),
        )

    def search(self):
        """
        Perform binary search to find the maximum QPS under the SLO
        """

        logger.info(
            f"Starting search. SLO type: {self.capacity_search_config.slo_type}, start QPS: {self.capacity_search_config.start_qps}",
        )

        left = 0
        right = self.capacity_search_config.start_qps * 2
        qps = 0
        last_qps = 0
        max_qps_under_sla = None
        min_qps_over_sla = 2**32

        tbt_at_max_qps = None
        ttft_at_max_qps = None
        tpot_at_max_qps = None
        deadline_miss_rate_at_max_qps = None
        best_run_id = None
        found_valid_qps = False

        for _ in range(self.capacity_search_config.max_iterations):
            logger.info(f"Searching between {left} and {right}")
            # stopping condition - we have reached the minimum granularity
            if (
                abs(left - right)
                < self.capacity_search_config.min_search_granularity * qps / 100
            ):
                break

            qps = (left + right) / 2
            # round to 2 decimal places
            qps = round(qps, 2)

            if qps == last_qps:
                break

            last_qps = qps

            (
                is_under_sla,
                tbt,
                ttft,
                tpot,
                deadline_miss_rate,
                run_id,
            ) = self._run_capacity_search_benchmark(qps)

            if is_under_sla:
                found_valid_qps = True
                max_qps_under_sla = qps
                tbt_at_max_qps = tbt
                ttft_at_max_qps = ttft
                tpot_at_max_qps = tpot
                deadline_miss_rate_at_max_qps = deadline_miss_rate
                best_run_id = run_id

                if qps > VICINITY_THRESHOLD * right:
                    right = min(right * QPS_INCREASE_SCALE, min_qps_over_sla)

                left = qps
            else:
                right = qps
                min_qps_over_sla = min(min_qps_over_sla, qps)

        if not found_valid_qps:
            logger.info(
                f"No valid QPS found.",
            )
            return {}

        logger.info(
            f"Max QPS under SLO: "
            f"QPS: {max_qps_under_sla}, "
            f"TBT P{self.capacity_search_config.tbt_percentile * 100}: {tbt_at_max_qps}, "
            f"TTFT P{self.capacity_search_config.ttft_percentile * 100}: {ttft_at_max_qps}, "
            f"TPOT P{self.capacity_search_config.tpot_percentile * 100}: {tpot_at_max_qps}, "
            f"Deadline Miss Rate P{self.capacity_search_config.deadline_miss_rate_percentile * 100}: {deadline_miss_rate_at_max_qps}"
            f"Best Run ID: {best_run_id} \n",
        )

        if (
            self.capacity_search_config.wandb_project is not None
            and self.capacity_search_config.enable_wandb_sweep
        ):
            best_run = wandb.Api().run(
                f"{self.capacity_search_config.wandb_project}/{best_run_id}"
            )
            best_run.tags.append("BEST_CONFIG")
            best_run.update()

        return {
            **self.capacity_search_config.to_dict(),
            "max_qps_under_sla": max_qps_under_sla,
            "deadline_miss_rate_at_max_qps": deadline_miss_rate_at_max_qps,
        }
