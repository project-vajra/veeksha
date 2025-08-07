import glob
import hashlib
import json
import os
import threading
from dataclasses import replace
from typing import Optional, Tuple

import wandb

from veeksha.capacity_search.benchmark_wrapper import run_benchmark_wrapped
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.config.utils import dataclass_to_dict
from veeksha.constants.capacity_search_constants import (
    QPS_INCREASE_SCALE,
    VICINITY_THRESHOLD,
)
from veeksha.logger import init_logger
from veeksha.capacity_search.slo_evaluator import SloEvaluator
from veeksha.capacity_search.slo import SloSet

logger = init_logger(__name__)


class CapacitySearch:
    def __init__(
        self,
        capacity_search_config: CapacitySearchConfig,
    ) -> None:
        self.capacity_search_config = capacity_search_config

        # will be cloned for each QPS attempt (changing output_dir, wandb_run_name)
        self.base_benchmark_config: BenchmarkConfig = (
            self.capacity_search_config.benchmark_config
        )

        self.stop_event = threading.Event()

        self.full_config = {
            "capacity_search_config": dataclass_to_dict(self.capacity_search_config),
            "benchmark_config": dataclass_to_dict(self.base_benchmark_config),
        }

        model_name = self.base_benchmark_config.client_config.model.split("/")[-1]
        config_hash = hashlib.md5(str(self.full_config).encode()).hexdigest()[:8]

        self.job_output_dir = os.path.join(
            self.capacity_search_config.output_dir, f"{model_name}_{config_hash}"
        )
        os.makedirs(self.job_output_dir, exist_ok=True)

        with open(os.path.join(self.job_output_dir, "config.json"), "w") as f:
            json.dump(self.full_config, f, indent=4)
        
        
        self.slo_set = SloSet(slos=self.capacity_search_config.slos)
        self.slo_evaluator = SloEvaluator(self.slo_set)

    def _build_benchmark_config_for_qps(
        self, qps: float, run_dir: str
    ) -> BenchmarkConfig:
        """Return a new BenchmarkConfig with metrics_config pointing to run_dir and
        wandb_run_name encoding QPS.
        """

        # copy of metric_config with updated output_dir and wandb_run_name
        new_metrics_cfg = replace(
            self.base_benchmark_config.metrics_config,
            output_dir=run_dir,
            wandb_run_name=f"qps_{qps}_model_{self.base_benchmark_config.client_config.model}",
        )

        # copy of benchmark_config with updated metrics_config
        return replace(self.base_benchmark_config, metrics_config=new_metrics_cfg)

    def _run_capacity_search_benchmark(
        self, qps: float
    ) -> Tuple[
        bool, Optional[float], Optional[float], Optional[float], Optional[float], str
    ]:
        qps_run_dir = os.path.join(self.job_output_dir, str(qps))

        # isolated benchmark config for this QPS
        benchmark_config = self._build_benchmark_config_for_qps(qps, qps_run_dir)

        cached_request_level_metrics_file = self._get_request_level_metrics(qps_run_dir)

        if cached_request_level_metrics_file is not None:
            logger.info(f"Cached results found for {qps}")
            return self._is_under_sla(cached_request_level_metrics_file, qps)

        run_benchmark_wrapped(benchmark_config)

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

    def _is_under_sla(
        self,
        request_level_metrics_file: str,
        qps: float,
    ) -> Tuple[
        bool, Optional[float], Optional[float], Optional[float], Optional[float], str
    ]:
        is_under_sla, metrics_dict = self.slo_evaluator.evaluate_request_metrics(
            request_level_metrics_file
        )
        
        print("METRICS_DICT------------------------", metrics_dict)
            
        logger.info(f"QPS: {qps} - {self.slo_evaluator.get_metrics_summary(metrics_dict)}")
        return (
            is_under_sla,
            metrics_dict.get(f"tbt_p{int(self.capacity_search_config.tbt_percentile * 100)}"),
            metrics_dict.get(f"ttft_p{int(self.capacity_search_config.ttft_percentile * 100)}"),
            metrics_dict.get(f"tpot_p{int(self.capacity_search_config.tpot_percentile * 100)}"),
            metrics_dict.get(f"deadline_miss_rate_p{int(self.capacity_search_config.deadline_miss_rate_percentile * 100)}"),
            str(qps),
        )

    def search(self):
        """
        Perform binary search to find the maximum QPS under the SLO
        """

        logger.info(
            f"Starting search. Start QPS: {self.capacity_search_config.start_qps}",
        )
        logger.info(f"SLOs: {self.slo_evaluator.slo_set}")

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

            qps = round((left + right) / 2, 2)

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
