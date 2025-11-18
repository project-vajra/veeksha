import glob
import json
import os
import tempfile
import threading
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, TypedDict, cast

import pandas as pd
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

MAX_NUM_THREADS = 72


class SearchResult(TypedDict, total=False):
    """Result of a capacity search."""

    max_buffer_size_under_sla: Optional[int]
    slo_metrics_at_max_buffer_size: Optional[Dict[str, float]]


class CapacitySearch:
    def __init__(
        self,
        capacity_search_config: CapacitySearchConfig,
    ) -> None:
        self.capacity_search_config = capacity_search_config

        # will be cloned for each buffer size attempt (changing output_dir, wandb_run_name)
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

    def _build_benchmark_config_for_buffer_size(
        self, buffer_size: int, run_dir: str
    ) -> BenchmarkConfig:
        """Return a new BenchmarkConfig with metrics_config.output_dir pointing to run_dir and
        wandb_run_name encoding buffer_size, max_concurrent_sessions set to buffer_size, and
        num_threads adjusted to min(num_threads, buffer_size).
        """

        # propagate wandb project from capacity search if provided and enable logging
        base_metrics_cfg = self.base_benchmark_config.metrics_config  # type: ignore
        propagated_project = (
            self.capacity_search_config.wandb_project
            if self.capacity_search_config.wandb_project is not None
            else base_metrics_cfg.wandb_project
        )
        enable_wandb = base_metrics_cfg.should_write_metrics_to_wandb or (
            self.capacity_search_config.wandb_project is not None
        )
        # effective group: capsearch override -> benchmark group -> auto group
        auto_group = (
            f"capsearch-{os.path.basename(self.job_output_dir)}"
            if self.job_output_dir
            else None
        )
        effective_group = (
            self.capacity_search_config.wandb_group
            or base_metrics_cfg.wandb_group
            or auto_group
        )

        # clone metrics config with updated output_dir and wandb settings
        new_metrics_cfg = replace(  # type: ignore[call-overload]
            cast(Any, base_metrics_cfg),
            output_dir=run_dir,
            wandb_run_name=f"buffer_{buffer_size}_model_{self.base_benchmark_config.client_config.model}",
            should_write_metrics_to_wandb=enable_wandb,
            wandb_project=propagated_project,
            wandb_group=effective_group,
        )

        original_num_threads = self.base_benchmark_config.num_request_runner_threads
        adjusted_num_threads = min(MAX_NUM_THREADS, max(original_num_threads, buffer_size))

        logger.info(
            f"Buffer size: {buffer_size}, Original num_threads: {original_num_threads}, Max num_threads: {MAX_NUM_THREADS} "
            f"Adjusted num_threads: {adjusted_num_threads}"
        )

        if adjusted_num_threads != original_num_threads:
            logger.info(
                f"Adjusted num_threads from {original_num_threads} to {adjusted_num_threads} "
                f"(limited by max_concurrent_sessions={buffer_size})"
            )

        # copy of benchmark_config with updated metrics, client_config, and max_concurrent_sessions
        return replace(  # type: ignore[call-overload]
            cast(Any, self.base_benchmark_config),
            metrics_config=new_metrics_cfg,
            max_concurrent_sessions=buffer_size,
            num_request_runner_threads=adjusted_num_threads,
        )

    def _ensure_run_dir(self) -> None:
        if self.job_output_dir is None:
            now = datetime.now()
            timestamp = (
                now.strftime("%Y%m%d-%H%M%S") + f"-{now.microsecond // 1000:03d}"
            )
            self.job_output_dir = os.path.join(self.job_root_dir, timestamp)
            os.makedirs(self.job_output_dir, exist_ok=True)
            with open(
                os.path.join(self.job_output_dir, "config.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(self.full_config, f, indent=4)

    def _run_capacity_search_benchmark(
        self, buffer_size: int
    ) -> Tuple[bool, Optional[Dict[str, float]], str, bool]:
        buffer_key = str(buffer_size)

        # cached_iter = self._capsearch_cache.get("iterations", {}).get(buffer_key)
        # if cached_iter is not None:
        #     logger.info(f"Using capacity search cache for buffer size {buffer_size}")
        #     return (
        #         bool(cached_iter.get("is_under_sla", False)),
        #         cached_iter.get("slo_metrics", {}),
        #         buffer_key,
        #         True,  # from_cache = True
        #     )

        # no cache: ensure per-run dir exists now
        self._ensure_run_dir()
        assert self.job_output_dir is not None
        buffer_run_dir = os.path.join(self.job_output_dir, str(buffer_size))
        os.makedirs(buffer_run_dir, exist_ok=True)

        # isolated benchmark config for this buffer size
        benchmark_config = self._build_benchmark_config_for_buffer_size(buffer_size, buffer_run_dir)

        service_metrics = run_benchmark_wrapped(benchmark_config)

        is_under_sla, slo_metrics_dict = self.slo_evaluator.evaluate_slo(
            service_metrics.metric_store
        )

        self._cache_iteration(
            buffer_size=buffer_key,
            is_under_sla=is_under_sla,
            slo_metrics=slo_metrics_dict,
            run_id=buffer_key,
        )

        return is_under_sla, slo_metrics_dict, buffer_key, False  # from_cache = False

    def _read_wandb_path_for_buffer_size(self, buffer_key: str) -> Optional[str]:
        """Read persisted wandb run path for a given buffer size attempt, if present."""
        try:
            if self.job_output_dir is None:
                return None
            target_dir = os.path.join(self.job_output_dir, str(buffer_key))
            run_info_path = os.path.join(target_dir, "wandb_run.json")
            if not os.path.exists(run_info_path):
                candidates = glob.glob(
                    os.path.join(target_dir, "**", "wandb_run.json"), recursive=True
                )
                if not candidates:
                    return None
                run_info_path = max(candidates, key=lambda p: os.path.getmtime(p))
            with open(run_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            entity = info.get("entity")
            project = info.get("project")
            run_id = info.get("id")
            return f"{entity}/{project}/{run_id}"
        except Exception:
            logger.debug(f"Could not read wandb path for buffer size {buffer_key}", exc_info=True)
            return None

    def _log_post_search_summary(self, benchmark_id: str) -> None:
        """Create a standalone wandb run with a buffer size vs SLO summary table/plot."""
        if self.capacity_search_config.wandb_project is None:
            return
        # build dataframe from cached iterations
        iterations = self._capsearch_cache.get("iterations", {})
        if len(iterations) == 0:
            return
        rows = []
        all_metric_keys: set[str] = set()
        for buffer_key, entry in iterations.items():
            slo_metrics = entry.get("slo_metrics", {}) or {}
            all_metric_keys.update(slo_metrics.keys())
        for buffer_key, entry in iterations.items():
            row: Dict[str, Any] = {"buffer_size": int(buffer_key)}
            slo_metrics = entry.get("slo_metrics", {}) or {}
            for k in all_metric_keys:
                row[k] = slo_metrics.get(k)
            rows.append(row)
        df = pd.DataFrame(sorted(rows, key=lambda r: r["buffer_size"]))

        # use the same effective group as attempts to visually group all runs
        auto_group = (
            f"capsearch-{os.path.basename(self.job_output_dir)}"
            if self.job_output_dir
            else None
        )
        effective_group = (
            self.capacity_search_config.wandb_group
            or getattr(self.base_benchmark_config.metrics_config, "wandb_group", None)
            or auto_group
        )

        run = wandb.init(
            project=self.capacity_search_config.wandb_project,
            group=effective_group,
            name=f"capsearch-buffered-summary-{benchmark_id}",
            config={
                "benchmark_id": benchmark_id,
                "model": self.base_benchmark_config.client_config.model,
                "start_buffer_size": self.capacity_search_config.start_qps,  # reusing start_qps config
                "max_iterations": self.capacity_search_config.max_iterations,
                "slos": str(self.slo_evaluator.slo_set),
            },
        )
        try:
            wandb.log({"capsearch_buffer_slo_table": wandb.Table(dataframe=df)}, step=0)
        finally:
            wandb.finish(quiet=True)

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

    def search(self) -> SearchResult:
        """
        Perform binary search to find the maximum buffer size under the SLO
        """

        logger.info(
            f"Starting search. Start buffer size: {self.capacity_search_config.start_qps}",  # reusing start_qps config
        )
        logger.info(f"SLOs: {self.slo_evaluator.slo_set}")

        # Emit effective wandb settings (if enabled)
        effective_project = self.capacity_search_config.wandb_project or getattr(
            self.base_benchmark_config.metrics_config, "wandb_project", None
        )
        effective_group = self.capacity_search_config.wandb_group or getattr(
            self.base_benchmark_config.metrics_config, "wandb_group", None
        )
        wandb_enabled = effective_project is not None or getattr(
            self.base_benchmark_config.metrics_config,
            "should_write_metrics_to_wandb",
            False,
        )
        if wandb_enabled:
            logger.info(
                f"wandb: enabled | project={effective_project} | group={effective_group}"
            )

        left = 1  # minimum buffer size of 1
        right = int(self.capacity_search_config.start_qps * 2)  # reusing start_qps config as start buffer size
        buffer_size = 0
        last_buffer_size = 0
        max_buffer_size_under_sla = None
        min_buffer_size_over_sla = 2**32

        slo_metrics_at_max_buffer_size = None
        best_run_id = None
        found_valid_buffer_size = False
        any_new_runs = False

        # Generate benchmark_id from base config for dashboard tracking
        import os

        benchmark_id = os.path.basename(
            self.base_benchmark_config.metrics_config.output_dir
        )

        # Emit start event
        from veeksha.dashboard.events import CapacitySearchEvent
        from veeksha.dashboard.handler import emit_dashboard_event

        emit_dashboard_event(
            CapacitySearchEvent(
                current_qps=0.0,
                is_under_sla=False,
                slo_metrics={},
                slo_target=str(self.slo_evaluator.slo_set),
                iteration=0,
                total_iterations=self.capacity_search_config.max_iterations,
                search_left=left,
                search_right=right,
                best_qps=None,
                best_slo_metrics=None,
                is_complete=False,
                benchmark_id=benchmark_id,
            )
        )

        # If qps_values is provided, use it as buffer_size_values (similar to how start_qps is reused)
        if self.capacity_search_config.qps_values is not None:
            logger.info(f"Running capacity search with specific buffer size values: {self.capacity_search_config.qps_values}")

            buffer_size_list = sorted([int(val) for val in self.capacity_search_config.qps_values])
            total_iterations = len(buffer_size_list)

            for iteration, buffer_size in enumerate(buffer_size_list):
                logger.info(f"Testing buffer size: {buffer_size} ({iteration + 1}/{total_iterations})")

                (
                    is_under_sla,
                    metrics_dict,
                    run_id,
                    from_cache,
                ) = self._run_capacity_search_benchmark(buffer_size)

                if not from_cache:
                    any_new_runs = True

                if is_under_sla:
                    found_valid_buffer_size = True
                    if max_buffer_size_under_sla is None or buffer_size > max_buffer_size_under_sla:
                        max_buffer_size_under_sla = buffer_size
                        slo_metrics_at_max_buffer_size = metrics_dict
                        best_run_id = run_id

                # Emit event after each iteration
                emit_dashboard_event(
                    CapacitySearchEvent(
                        current_qps=float(buffer_size),
                        is_under_sla=is_under_sla,
                        slo_metrics=metrics_dict or {},
                        slo_target=str(self.slo_evaluator.slo_set),
                        iteration=iteration + 1,
                        total_iterations=total_iterations,
                        search_left=min(buffer_size_list),
                        search_right=max(buffer_size_list),
                        best_qps=float(max_buffer_size_under_sla) if max_buffer_size_under_sla else None,
                        best_slo_metrics=slo_metrics_at_max_buffer_size,
                        is_complete=False,
                        from_cache=from_cache,
                        benchmark_id=benchmark_id,
                    )
                )

            # Skip binary search and jump to results logging
        else:
            # Original binary search logic
            for iteration in range(self.capacity_search_config.max_iterations):
                logger.info(f"Searching between {left} and {right}")
                # stopping condition - we have reached the minimum granularity (buffer sizes are integers)
                if abs(left - right) <= 1:
                    break

                buffer_size = int((left + right) / 2)

                if buffer_size == last_buffer_size:
                    break

                last_buffer_size = buffer_size

                (
                    is_under_sla,
                    metrics_dict,
                    run_id,
                    from_cache,
                ) = self._run_capacity_search_benchmark(buffer_size)

                if not from_cache:
                    any_new_runs = True

                if is_under_sla:
                    found_valid_buffer_size = True
                    max_buffer_size_under_sla = buffer_size
                    slo_metrics_at_max_buffer_size = metrics_dict
                    best_run_id = run_id

                    # For buffer sizes, if we're near the top, expand search range
                    if buffer_size > VICINITY_THRESHOLD * right:
                        right = min(int(right * QPS_INCREASE_SCALE), min_buffer_size_over_sla)

                    left = buffer_size
                else:
                    right = buffer_size
                    min_buffer_size_over_sla = min(min_buffer_size_over_sla, buffer_size)

                # Emit event after each iteration
                emit_dashboard_event(
                    CapacitySearchEvent(
                        current_qps=float(buffer_size),  # using qps field for buffer_size for compatibility
                        is_under_sla=is_under_sla,
                        slo_metrics=metrics_dict or {},
                        slo_target=str(self.slo_evaluator.slo_set),
                        iteration=iteration + 1,
                        total_iterations=self.capacity_search_config.max_iterations,
                        search_left=left,
                        search_right=right,
                        best_qps=float(max_buffer_size_under_sla) if max_buffer_size_under_sla else None,  # using qps field for buffer_size
                        best_slo_metrics=slo_metrics_at_max_buffer_size,
                        is_complete=False,
                        from_cache=from_cache,
                        benchmark_id=benchmark_id,
                    )
                )

        if not found_valid_buffer_size:
            logger.info(
                f"No valid buffer size found.",
            )
            return {}

        logger.info(
            f"{'-'*100}\n"
            f"Max buffer size found by Capacity Search with: \n"
            f"    * SLOs: {self.slo_evaluator.slo_set} \n"
            f"    * SLO Metrics: {slo_metrics_at_max_buffer_size} \n"
            f"    * Best Run ID: {best_run_id} \n"
            f"is {max_buffer_size_under_sla} \n"
            f"{'-'*100}\n"
        )

        if any_new_runs and wandb_enabled and best_run_id is not None:
            best_path = self._read_wandb_path_for_buffer_size(str(best_run_id))
            if best_path:
                try:
                    api = wandb.Api()
                    best_run = api.run(best_path)
                    current = list(best_run.tags or [])
                    if "BEST_CONFIG" not in current:
                        best_run.tags = current + ["BEST_CONFIG"]
                        best_run.update()
                except Exception:
                    logger.warning(
                        "Failed to tag BEST_CONFIG for run %s", best_path, exc_info=True
                    )

        self._cache_final(
            max_buffer_size_under_sla=max_buffer_size_under_sla,
            slo_metrics_at_max_buffer_size=slo_metrics_at_max_buffer_size,
            best_run_id=best_run_id,
        )

        # Emit final completion event
        emit_dashboard_event(
            CapacitySearchEvent(
                current_qps=float(max_buffer_size_under_sla) if max_buffer_size_under_sla else 0.0,
                is_under_sla=True,
                slo_metrics=slo_metrics_at_max_buffer_size or {},
                slo_target=str(self.slo_evaluator.slo_set),
                iteration=self.capacity_search_config.max_iterations,
                total_iterations=self.capacity_search_config.max_iterations,
                search_left=left,
                search_right=right,
                best_qps=float(max_buffer_size_under_sla) if max_buffer_size_under_sla else None,
                best_slo_metrics=slo_metrics_at_max_buffer_size,
                is_complete=True,
                benchmark_id=benchmark_id,
            )
        )

        # Log a post-search summary run/table only if new runs occurred
        if any_new_runs:
            self._log_post_search_summary(benchmark_id)

        return {
            "max_buffer_size_under_sla": max_buffer_size_under_sla,
            "slo_metrics_at_max_buffer_size": slo_metrics_at_max_buffer_size,
        }

    def _load_cache(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self._capsearch_cache_file):
                with open(self._capsearch_cache_file, "r", encoding="utf-8") as f:
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
            target_path = self._capsearch_cache_file
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)

            fd, tmp_path = tempfile.mkstemp(
                prefix="._capsearch_cache.", suffix=".json", dir=target_dir
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._capsearch_cache, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, target_path)
            finally:
                # If replace failed, ensure temp file is removed
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except FileNotFoundError:
                        pass
        except Exception as e:
            logger.warning(f"Failed to write capsearch cache: {e}")

    def _cache_iteration(
        self,
        buffer_size: str,
        is_under_sla: bool,
        slo_metrics: Dict[str, float],
        run_id: str,
    ) -> None:
        if "iterations" not in self._capsearch_cache:
            self._capsearch_cache["iterations"] = {}
        self._capsearch_cache["iterations"][buffer_size] = {
            "is_under_sla": is_under_sla,
            "slo_metrics": slo_metrics,
            "run_id": run_id,
        }
        self._save_cache()

    def _cache_final(
        self,
        max_buffer_size_under_sla: Optional[int],
        slo_metrics_at_max_buffer_size: Optional[Dict[str, float]],
        best_run_id: Optional[str],
    ) -> None:
        self._capsearch_cache["final"] = {
            "max_buffer_size_under_sla": max_buffer_size_under_sla,
            "slo_metrics_at_max_buffer_size": slo_metrics_at_max_buffer_size,
            "best_run_id": best_run_id,
            "slos": str(self.slo_evaluator.slo_set),
        }
        self._save_cache()
