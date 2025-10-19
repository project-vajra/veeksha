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
from veeksha.config.generators.request_generator.base_generator import (
    BaseRequestGeneratorConfig,
)
from veeksha.config.generators.request_generator.lmeval_generator import (
    LmevalRequestGeneratorConfig,
)
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.config.generators.request_generator.trace_generator import (
    TraceRequestGeneratorConfig,
)
from veeksha.config.utils import dataclass_to_dict, get_config_hash
from veeksha.constants.capacity_search_constants import (
    QPS_INCREASE_SCALE,
    VICINITY_THRESHOLD,
)
from veeksha.logger import init_logger

logger = init_logger(__name__)


class SearchResult(TypedDict, total=False):
    """Result of a capacity search."""

    max_qps_under_sla: Optional[float]
    slo_metrics_at_max_qps: Optional[Dict[str, float]]


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
        """Return a new BenchmarkConfig with metrics_config.output_dir pointing to run_dir and
        wandb_run_name encoding QPS.
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
            wandb_run_name=f"qps_{qps}_model_{self.base_benchmark_config.client_config.model}",
            should_write_metrics_to_wandb=enable_wandb,
            wandb_project=propagated_project,
            wandb_group=effective_group,
        )
        # Build a request_generator_config adjusted for this attempt's QPS
        # using dataclasses.replace to respect frozen dataclasses.
        new_req_gen_cfg = self._apply_qps_to_request_generator_config(
            self.base_benchmark_config.request_generator_config, qps
        )

        if new_req_gen_cfg is self.base_benchmark_config.request_generator_config:
            logger.warning(
                f"QPS override (qps={qps}) not applied to request generator type {type(self.base_benchmark_config.request_generator_config).__name__}; run will use the base request rate.",
            )

        # copy of benchmark_config with updated metrics and request generator config
        return replace(  # type: ignore[call-overload]
            cast(Any, self.base_benchmark_config),
            metrics_config=new_metrics_cfg,
            request_generator_config=new_req_gen_cfg,  # type: ignore[arg-type]
        )

    def _apply_qps_to_request_generator_config(
        self,
        base_req_gen_cfg: BaseRequestGeneratorConfig,
        qps: float,
    ) -> BaseRequestGeneratorConfig:
        """Return a copy of request generator config with QPS applied.

        - Synthetic: set interval_generator_config.qps
        - Trace + sessions: set session_interval_generator_config.qps
        - LMEval: set interval_generator_config.qps
        """
        new_req_gen_cfg: BaseRequestGeneratorConfig = base_req_gen_cfg

        if isinstance(base_req_gen_cfg, SyntheticRequestGeneratorConfig):
            interval_cfg = base_req_gen_cfg.interval_generator_config
            if hasattr(interval_cfg, "qps"):
                new_interval_cfg = replace(cast(Any, interval_cfg), qps=qps)  # type: ignore[call-overload]
                new_req_gen_cfg = replace(  # type: ignore[call-overload]
                    cast(Any, base_req_gen_cfg),
                    interval_generator_config=new_interval_cfg,
                )
        elif isinstance(base_req_gen_cfg, LmevalRequestGeneratorConfig):
            interval_cfg = base_req_gen_cfg.interval_generator_config
            if hasattr(interval_cfg, "qps"):
                new_interval_cfg = replace(cast(Any, interval_cfg), qps=qps)  # type: ignore[call-overload]
                new_req_gen_cfg = replace(  # type: ignore[call-overload]
                    cast(Any, base_req_gen_cfg),
                    interval_generator_config=new_interval_cfg,
                )
        elif isinstance(base_req_gen_cfg, TraceRequestGeneratorConfig):
            session_gen_cfg = base_req_gen_cfg.session_generator_config
            if session_gen_cfg is not None:
                session_interval_cfg = session_gen_cfg.session_interval_generator_config
                if session_interval_cfg is not None and hasattr(
                    session_interval_cfg, "qps"
                ):
                    new_session_interval_cfg = replace(  # type: ignore[call-overload]
                        cast(Any, session_interval_cfg), qps=qps
                    )
                    new_session_gen_cfg = replace(  # type: ignore[call-overload]
                        cast(Any, session_gen_cfg),
                        session_interval_generator_config=new_session_interval_cfg,
                    )
                    new_req_gen_cfg = replace(  # type: ignore[call-overload]
                        cast(Any, new_req_gen_cfg),
                        session_generator_config=new_session_gen_cfg,
                    )
                    logger.info(
                        f"Capacity search: detected session trace generator; interpreting qps={qps} as sessions per second (SPS)."
                    )

        return new_req_gen_cfg

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
        self, qps: float
    ) -> Tuple[bool, Optional[Dict[str, float]], str, bool]:
        qps_key = str(qps)

        cached_iter = self._capsearch_cache.get("iterations", {}).get(qps_key)
        if cached_iter is not None:
            logger.info(f"Using capacity search cache for QPS {qps}")
            return (
                bool(cached_iter.get("is_under_sla", False)),
                cached_iter.get("slo_metrics", {}),
                qps_key,
                True,  # from_cache = True
            )

        # no cache: ensure per-run dir exists now
        self._ensure_run_dir()
        assert self.job_output_dir is not None
        qps_run_dir = os.path.join(self.job_output_dir, str(qps))
        os.makedirs(qps_run_dir, exist_ok=True)

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

        return is_under_sla, slo_metrics_dict, qps_key, False  # from_cache = False

    def _read_wandb_path_for_qps(self, qps_key: str) -> Optional[str]:
        """Read persisted wandb run path for a given QPS attempt, if present."""
        try:
            if self.job_output_dir is None:
                return None
            target_dir = os.path.join(self.job_output_dir, str(qps_key))
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
            logger.debug(f"Could not read wandb path for QPS {qps_key}", exc_info=True)
            return None

    def _log_post_search_summary(self, benchmark_id: str) -> None:
        """Create a standalone wandb run with a QPS vs SLO summary table/plot."""
        if self.capacity_search_config.wandb_project is None:
            return
        # build dataframe from cached iterations
        iterations = self._capsearch_cache.get("iterations", {})
        if len(iterations) == 0:
            return
        rows = []
        all_metric_keys: set[str] = set()
        for qps_key, entry in iterations.items():
            slo_metrics = entry.get("slo_metrics", {}) or {}
            all_metric_keys.update(slo_metrics.keys())
        for qps_key, entry in iterations.items():
            row: Dict[str, Any] = {"qps": float(qps_key)}
            slo_metrics = entry.get("slo_metrics", {}) or {}
            for k in all_metric_keys:
                row[k] = slo_metrics.get(k)
            rows.append(row)
        df = pd.DataFrame(sorted(rows, key=lambda r: r["qps"]))

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
            name=f"capsearch-summary-{benchmark_id}",
            config={
                "benchmark_id": benchmark_id,
                "model": self.base_benchmark_config.client_config.model,
                "start_qps": self.capacity_search_config.start_qps,
                "max_iterations": self.capacity_search_config.max_iterations,
                "slos": str(self.slo_evaluator.slo_set),
            },
        )
        try:
            wandb.log({"capsearch_qps_slo_table": wandb.Table(dataframe=df)}, step=0)
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
        Perform binary search to find the maximum QPS under the SLO
        """

        logger.info(
            f"Starting search. Start QPS: {self.capacity_search_config.start_qps}",
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

        left = 0
        right = self.capacity_search_config.start_qps * 2
        qps = 0
        last_qps = 0
        max_qps_under_sla = None
        min_qps_over_sla = 2**32

        slo_metrics_at_max_qps = None
        best_run_id = None
        found_valid_qps = False
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

        for iteration in range(self.capacity_search_config.max_iterations):
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
                from_cache,
            ) = self._run_capacity_search_benchmark(qps)

            if not from_cache:
                any_new_runs = True

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

            # Emit event after each iteration
            emit_dashboard_event(
                CapacitySearchEvent(
                    current_qps=qps,
                    is_under_sla=is_under_sla,
                    slo_metrics=metrics_dict or {},
                    slo_target=str(self.slo_evaluator.slo_set),
                    iteration=iteration + 1,
                    total_iterations=self.capacity_search_config.max_iterations,
                    search_left=left,
                    search_right=right,
                    best_qps=max_qps_under_sla,
                    best_slo_metrics=slo_metrics_at_max_qps,
                    is_complete=False,
                    from_cache=from_cache,
                    benchmark_id=benchmark_id,
                )
            )

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

        if any_new_runs and wandb_enabled and best_run_id is not None:
            best_path = self._read_wandb_path_for_qps(str(best_run_id))
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
            max_qps_under_sla=max_qps_under_sla,
            slo_metrics_at_max_qps=slo_metrics_at_max_qps,
            best_run_id=best_run_id,
        )

        # Emit final completion event
        emit_dashboard_event(
            CapacitySearchEvent(
                current_qps=max_qps_under_sla or 0.0,
                is_under_sla=True,
                slo_metrics=slo_metrics_at_max_qps or {},
                slo_target=str(self.slo_evaluator.slo_set),
                iteration=self.capacity_search_config.max_iterations,
                total_iterations=self.capacity_search_config.max_iterations,
                search_left=left,
                search_right=right,
                best_qps=max_qps_under_sla,
                best_slo_metrics=slo_metrics_at_max_qps,
                is_complete=True,
                benchmark_id=benchmark_id,
            )
        )

        # Log a post-search summary run/table only if new runs occurred
        if any_new_runs:
            self._log_post_search_summary(benchmark_id)

        return {
            "max_qps_under_sla": max_qps_under_sla,
            "slo_metrics_at_max_qps": slo_metrics_at_max_qps,
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
