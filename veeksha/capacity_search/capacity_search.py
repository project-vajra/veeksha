import glob
import json
import os
import threading
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import wandb

from veeksha.capacity_search.benchmark_wrapper import run_benchmark_wrapped
from veeksha.capacity_search.slo import SloSet
from veeksha.capacity_search.slo_evaluator import SloEvaluator
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.config.utils import dataclass_to_dict, get_config_hash
from veeksha.constants.capacity_search_constants import (
    QPS_INCREASE_SCALE,
    VICINITY_THRESHOLD,
)
from veeksha.logger import init_logger

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

        config_hash = get_config_hash(self.full_config)

        # stable root dir to persist cache across runs
        self.job_root_dir = os.path.join(
            self.capacity_search_config.output_dir, f"{model_name}-{config_hash}"
        )
        os.makedirs(self.job_root_dir, exist_ok=True)

        # avoid empty dirs on cache hits
        self.job_output_dir = None

        self.slo_set = SloSet(slos=self.capacity_search_config.slos)
        self.slo_evaluator = SloEvaluator(self.slo_set)
        # can be reused across runs
        self._capsearch_cache_file = os.path.join(
            self.job_root_dir, "_capsearch_cache.json"
        )
        self._capsearch_cache = self._load_cache()

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

    def _ensure_run_dir(self) -> None:
        if self.job_output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.job_output_dir = os.path.join(self.job_root_dir, timestamp)
            os.makedirs(self.job_output_dir, exist_ok=True)
            with open(os.path.join(self.job_output_dir, "config.json"), "w") as f:
                json.dump(self.full_config, f, indent=4)

    def _run_capacity_search_benchmark(
        self, qps: float
    ) -> Tuple[bool, Optional[Dict[str, float]], str]:
        qps_key = str(qps)

        cached_iter = self._capsearch_cache.get("iterations", {}).get(qps_key)
        if cached_iter is not None:
            logger.info(f"Using capacity search cache for QPS {qps}")
            return (
                bool(cached_iter.get("is_under_sla", False)),
                cached_iter.get("slo_metrics", {}),
                qps_key,
            )

        # no cache: ensure per-run dir exists now
        self._ensure_run_dir()
        assert self.job_output_dir is not None
        qps_run_dir = os.path.join(self.job_output_dir, str(qps))

        # isolated benchmark config for this QPS
        benchmark_config = self._build_benchmark_config_for_qps(qps, qps_run_dir)

        service_metrics = run_benchmark_wrapped(benchmark_config)

        is_under_sla, slo_metrics_dict = self.slo_evaluator.evaluate_slo(
            service_metrics.metric_store
        )

        self._cache_iteration(
            qps=qps_key,
            is_under_sla=is_under_sla,
            slo_metrics=slo_metrics_dict,
            run_id=qps_key,
        )

        return is_under_sla, slo_metrics_dict, qps_key

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
    ) -> Tuple[bool, Optional[Dict[str, float]], str]:
        # user provided slos, percentiles and thresholds
        is_under_sla, slo_metrics_dict = (
            self.slo_evaluator.evaluate_slo_request_metrics(request_level_metrics_file)
        )

        return (
            is_under_sla,
            slo_metrics_dict,
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

        slo_metrics_at_max_qps = None
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
                metrics_dict,
                run_id,
            ) = self._run_capacity_search_benchmark(qps)

            if is_under_sla:
                found_valid_qps = True
                max_qps_under_sla = qps
                slo_metrics_at_max_qps = metrics_dict
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
            f"{'-'*100}\n"
            f"Max QPS found by Capacity Search with: \n"
            f"    * SLOs: {self.slo_evaluator.slo_set} \n"
            f"    * SLO Metrics: {slo_metrics_at_max_qps} \n"
            f"    * Best Run ID: {best_run_id} \n"
            f"is {max_qps_under_sla} \n"
            f"{'-'*100}\n"
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

        self._cache_final(
            max_qps_under_sla=max_qps_under_sla,
            slo_metrics_at_max_qps=slo_metrics_at_max_qps,
            best_run_id=best_run_id,
        )

        return {
            "max_qps_under_sla": max_qps_under_sla,
            "slo_metrics_at_max_qps": slo_metrics_at_max_qps,
        }

    def _load_cache(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self._capsearch_cache_file):
                with open(self._capsearch_cache_file, "r") as f:
                    cache = json.load(f)
                    return cache
        except Exception as e:
            logger.warning(f"Failed to read capsearch cache: {e}")
        return {
            "slos": str(self.slo_evaluator.slo_set),
            "iterations": {},
            "final": None,
        }

    def _save_cache(self) -> None:
        try:
            with open(self._capsearch_cache_file, "w") as f:
                json.dump(self._capsearch_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write capsearch cache: {e}")

    def _cache_iteration(
        self,
        qps: str,
        is_under_sla: bool,
        slo_metrics: Dict[str, float],
        run_id: str,
    ) -> None:
        if "iterations" not in self._capsearch_cache:
            self._capsearch_cache["iterations"] = {}
        self._capsearch_cache["iterations"][qps] = {
            "is_under_sla": is_under_sla,
            "slo_metrics": slo_metrics,
            "run_id": run_id,
        }
        self._save_cache()

    def _cache_final(
        self,
        max_qps_under_sla: Optional[float],
        slo_metrics_at_max_qps: Optional[Dict[str, float]],
        best_run_id: Optional[str],
    ) -> None:
        self._capsearch_cache["final"] = {
            "max_qps_under_sla": max_qps_under_sla,
            "slo_metrics_at_max_qps": slo_metrics_at_max_qps,
            "best_run_id": best_run_id,
            "slos": str(self.slo_evaluator.slo_set),
        }
        self._save_cache()
