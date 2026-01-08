"""Capacity search runs a benchmark multiple times with the same configuration,
varying only a single traffic scheduler knob. It then finds the maximum value
that still satisfies all configured SLOs, as recorded in `metrics/slo_results.json`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple, cast

import yaml

from veeksha.benchmark import manage_benchmark_run
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.config.generator.interval import (
    BaseIntervalGeneratorConfig,
    FixedIntervalGeneratorConfig,
)
from veeksha.config.traffic import ConcurrentTrafficConfig, RateTrafficConfig
from veeksha.config.utils import dataclass_to_dict
from veeksha.logger import init_logger
from veeksha.wandb_integration import (
    dedup_tags,
    maybe_log_capacity_search_summary,
    update_run_tags,
)

logger = init_logger(__name__)


def _persist_capacity_search_config_yaml(config: CapacitySearchConfig) -> str:
    """Write the resolved capacity search configuration to config.yml."""
    os.makedirs(config.output_dir, exist_ok=True)
    config_dict = dataclass_to_dict(config)
    config_path = os.path.join(config.output_dir, "config.yml")
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(
            config_dict,
            config_file,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    return config_path


def _init_capacity_search_output_dir(config: CapacitySearchConfig) -> str:
    """Resolve and prepare the final capacity search output directory.

    Mirrors `veeksha.new.benchmark_utils._init_output_dir` semantics:
    - persist config to YAML
    - hash the persisted config
    - create a timestamp+hash subdir inside the user-specified base output dir
    - move config.yml into the resolved dir
    - update `config.output_dir` in-place
    """
    base_output_dir = config.output_dir
    os.makedirs(base_output_dir, exist_ok=True)

    config_path = _persist_capacity_search_config_yaml(config)
    with open(config_path, "rb") as config_file:
        config_bytes = config_file.read()
    config_hash = hashlib.sha1(config_bytes).hexdigest()[:8]

    timestamp_prefix = datetime.now(timezone.utc).strftime("%d:%m:%Y-%H:%M:%S")
    base_dir_name = f"{timestamp_prefix}-{config_hash}"
    resolved_output_dir = os.path.join(base_output_dir, base_dir_name)

    suffix = 1
    while os.path.exists(resolved_output_dir):
        suffix += 1
        resolved_output_dir = os.path.join(base_output_dir, f"{base_dir_name}-{suffix}")

    os.makedirs(resolved_output_dir, exist_ok=True)
    shutil.move(config_path, os.path.join(resolved_output_dir, "config.yml"))
    object.__setattr__(config, "output_dir", resolved_output_dir)
    logger.info("Capacity search outputs will be stored in %s", resolved_output_dir)
    return resolved_output_dir


def _read_slo_results(run_output_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_output_dir, "metrics", "slo_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing SLO results at '{path}'. Ensure the benchmark evaluator writes "
            "request-level metrics and has at least one configured SLO."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid slo_results.json at '{path}': expected dict.")
    return cast(Dict[str, Any], data)


def _knob_description(benchmark_config: BenchmarkConfig) -> Tuple[str, str]:
    traffic = benchmark_config.traffic_scheduler
    if isinstance(traffic, RateTrafficConfig):
        return ("rate", "traffic_scheduler.interval_generator.arrival_rate")
    if isinstance(traffic, ConcurrentTrafficConfig):
        return ("concurrent", "traffic_scheduler.target_concurrent_sessions")
    raise ValueError(
        f"Unsupported traffic scheduler type: {type(traffic).__name__}. "
        "Supported: rate, concurrent."
    )


def patch_traffic_knob(
    benchmark_config: BenchmarkConfig, *, value: float
) -> BenchmarkConfig:
    """Return a copy of BenchmarkConfig with traffic knob set to the provided value."""
    traffic = benchmark_config.traffic_scheduler

    if isinstance(traffic, RateTrafficConfig):
        interval_cfg: BaseIntervalGeneratorConfig = traffic.interval_generator
        if isinstance(interval_cfg, FixedIntervalGeneratorConfig):
            # `value` as sessions-per-second and convert to fixed interval.
            new_interval_cfg = replace(interval_cfg, interval=float(1.0 / value))
        else:
            new_interval_cfg = replace(
                cast(Any, interval_cfg), arrival_rate=float(value)
            )
        new_traffic = replace(traffic, interval_generator=new_interval_cfg)
        return replace(benchmark_config, traffic_scheduler=new_traffic)

    if isinstance(traffic, ConcurrentTrafficConfig):
        if abs(value - round(value)) > 1e-9:
            raise ValueError(
                f"Concurrent capacity search requires integer values; got {value}."
            )
        target = int(round(value))
        # concurrent traffic ramps at ~1 session/sec
        new_traffic = replace(
            traffic,
            target_concurrent_sessions=target,
            rampup_seconds=target,
        )
        return replace(benchmark_config, traffic_scheduler=new_traffic)

    raise ValueError(
        f"Unsupported traffic scheduler type: {type(traffic).__name__}. "
        "Supported: rate, concurrent."
    )


def _adaptive_capacity_search(
    *,
    start_value: float,
    max_value: float,
    expansion_factor: float,
    is_passing: Callable[[float], bool],
    max_iterations: int,
    precision: int,
    integer_domain: bool,
) -> Tuple[Optional[float], int]:
    """Adaptive two-phase search: exponential probe then binary search."""
    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")
    if start_value <= 0:
        raise ValueError("start_value must be > 0")
    if expansion_factor <= 1.0:
        raise ValueError("expansion_factor must be > 1.0")
    if start_value > max_value:
        raise ValueError("start_value must be <= max_value")

    def _round_value(v: float) -> float:
        if integer_domain:
            return float(max(1, int(round(v))))
        scale = 10**precision
        return round(v * scale) / scale

    iters = 0
    probe = _round_value(start_value)
    last_passing: Optional[float] = None
    first_failing: Optional[float] = None

    # exponential probing
    logger.info("Phase 1: Probing from %s (expansion=%s)", probe, expansion_factor)

    while iters < max_iterations:
        iters += 1
        if is_passing(probe):
            last_passing = probe
            if probe >= max_value:
                logger.info("Passed at max_value=%s", max_value)
                return (last_passing, iters)
            next_probe = _round_value(probe * expansion_factor)
            if next_probe <= probe:
                next_probe = (
                    probe + 1 if integer_domain else probe + (1.0 / 10**precision)
                )
            probe = min(_round_value(next_probe), _round_value(max_value))
        else:
            first_failing = probe
            break

    if last_passing is None:
        logger.warning("Start value %s failed", start_value)
        return (None, iters)

    if first_failing is None:
        return (last_passing, iters)

    # binary search
    logger.info("Phase 2: Binary search in [%s, %s]", last_passing, first_failing)

    if integer_domain:
        lo = int(round(last_passing))
        hi = int(round(first_failing))
        best = lo

        while lo < hi and iters < max_iterations:
            mid = (lo + hi + 1) // 2
            iters += 1
            if is_passing(float(mid)):
                best = mid
                lo = mid
            else:
                hi = mid - 1

        return (float(best), iters)

    else:
        scale = 10**precision
        lo = int(round(last_passing * scale))
        hi = int(round(first_failing * scale))
        best_i = lo

        while lo < hi and iters < max_iterations:
            mid_i = (lo + hi + 1) // 2
            mid = mid_i / scale
            iters += 1
            if is_passing(mid):
                best_i = mid_i
                lo = mid_i
            else:
                hi = mid_i - 1

        return (best_i / scale, iters)


def run_capacity_search(config: CapacitySearchConfig) -> Dict[str, Any]:
    """Run capacity search and return a machine-readable result dict."""
    _init_capacity_search_output_dir(config)
    runs_dir = os.path.join(config.output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    base_benchmark_config = config.benchmark_config
    scheduler_type, knob_path = _knob_description(base_benchmark_config)

    os.makedirs(config.output_dir, exist_ok=True)

    iterations: list[Dict[str, Any]] = []
    attempt_counter = 0

    logger.info(
        "Capacity search: knob=%s | start=%s, max=%s, expansion=%s | max_iterations=%s | output_dir=%s",
        knob_path,
        config.start_value,
        config.max_value,
        config.expansion_factor,
        config.max_iterations,
        config.output_dir,
    )

    def run_one(value: float) -> bool:
        nonlocal attempt_counter
        attempt_counter += 1
        separator = "-" * 88
        logger.info(separator)
        logger.info(
            "Capacity search attempt %d | %s=%s",
            attempt_counter,
            knob_path,
            value,
        )
        # ensure run outputs are grouped under capacity search output_dir
        run_cfg = replace(base_benchmark_config, output_dir=runs_dir)
        run_cfg = patch_traffic_knob(run_cfg, value=value)

        # Group all attempts under one wandb group and give them readable names.
        if getattr(run_cfg, "wandb", None) and run_cfg.wandb.enabled:
            auto_group = f"capsearch-{os.path.basename(config.output_dir.rstrip('/'))}"
            short_knob = knob_path.rsplit(".", 1)[-1]
            run_cfg = replace(
                run_cfg,
                wandb=replace(
                    run_cfg.wandb,
                    group=run_cfg.wandb.group or auto_group,
                    run_name=run_cfg.wandb.run_name
                    or f"{attempt_counter:02d}-{short_knob}={value}",
                    tags=dedup_tags([*run_cfg.wandb.tags, "capsearch"]),
                ),
            )
        manage_benchmark_run(run_cfg)

        slo = _read_slo_results(run_cfg.output_dir)
        all_met = bool(slo.get("all_slos_met", False))

        slo_summaries = []
        for r in slo.get("results", []) if isinstance(slo.get("results"), list) else []:
            if not isinstance(r, dict):
                continue
            name = r.get("name") or r.get("slo_metric_key") or "slo"
            met = r.get("met")
            observed = r.get("observed_value")
            threshold = r.get("threshold")
            if observed is None or threshold is None:
                slo_summaries.append(f"{name}={met}")
            else:
                slo_summaries.append(f"{name}:{observed:.4f}<={threshold:.4f}={met}")

        logger.info(
            "Capacity search attempt %d result: %s | run_dir=%s%s",
            attempt_counter,
            "PASS" if all_met else "FAIL",
            run_cfg.output_dir,
            (f" | {' | '.join(slo_summaries)}" if slo_summaries else ""),
        )
        iterations.append(
            {
                "value": value,
                "all_slos_met": all_met,
                "run_dir": run_cfg.output_dir,
                "slo_results": slo,
            }
        )
        return all_met

    integer_domain = scheduler_type == "concurrent"

    best_value, iters_used = _adaptive_capacity_search(
        start_value=config.start_value,
        max_value=config.max_value,
        expansion_factor=config.expansion_factor,
        is_passing=run_one,
        max_iterations=config.max_iterations,
        precision=config.precision,
        integer_domain=integer_domain,
    )

    # If we hit the search budget, we may not have proven the true maximum.
    if (
        iters_used >= config.max_iterations
        and best_value is not None
        and best_value != config.max_value
    ):
        logger.warning(
            "Capacity search stopped after max_iterations=%s without reaching max_value=%s; "
            "current best=%s. Increase max_iterations to fully search the range.",
            config.max_iterations,
            config.max_value,
            best_value,
        )

    best_run_dir = None
    if best_value is not None:
        # last passing run for the best value
        for entry in reversed(iterations):
            if entry["value"] == best_value and entry["all_slos_met"]:
                best_run_dir = entry["run_dir"]
                break

    result: Dict[str, Any] = {
        "traffic_scheduler_type": scheduler_type,
        "searched_knob": knob_path,
        "start_value": config.start_value,
        "max_value": config.max_value,
        "expansion_factor": config.expansion_factor,
        "max_iterations": config.max_iterations,
        "precision": config.precision,
        "iterations_run": iters_used,
        "best_value": best_value,
        "best_run_dir": best_run_dir,
        "history": iterations,
    }

    out_path = os.path.join(config.output_dir, "capacity_search_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Capacity search result written to %s", out_path)
    if best_value is None:
        logger.warning(
            "Capacity search complete: no passing value found starting from %s for %s",
            config.start_value,
            knob_path,
        )
    else:
        logger.info(
            "Capacity search best: %s=%s | run_dir=%s",
            knob_path,
            best_value,
            best_run_dir,
        )
        if best_run_dir:
            update_run_tags(best_run_dir, ["BEST_CONFIG"])

    bench_cfg = config.benchmark_config
    if getattr(bench_cfg, "wandb", None) and bench_cfg.wandb.enabled:
        auto_group = f"capsearch-{os.path.basename(config.output_dir.rstrip('/'))}"
        maybe_log_capacity_search_summary(
            output_dir=config.output_dir,
            wandb_cfg=bench_cfg.wandb,
            result=result,
            group=bench_cfg.wandb.group or auto_group,
        )
    return result


def main() -> None:
    configs = CapacitySearchConfig.create_from_cli_args()
    for cfg in configs:
        logger.info("Running capacity search (output_dir=%s)", cfg.output_dir)
        run_capacity_search(cfg)


if __name__ == "__main__":
    main()
