"""
This module centralizes:
- `wandb.init()` / `wandb.finish()` lifecycle per benchmark run
- logging of key scalar metrics
- uploading common run artifacts (config, metrics, plots, health checks)
- persisting `wandb_run.json` in the run output directory

Evaluators may still log additional plots/tables opportunistically when
`wandb.run` exists, but they should not own wandb configuration.
"""

from __future__ import annotations

import json
import os
from csv import DictReader, DictWriter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, cast

from vidhi import dataclass_to_dict

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.wandb import WandbConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "access_token",
    "token",
    "secret",
    "password",
}


def dedup_tags(tags: Iterable[str]) -> list[str]:
    """Deduplicate tags while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        t = str(t).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _scrub_secrets(obj: Any) -> Any:
    """Recursively redact likely-secret values from dict-like structures."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _SENSITIVE_KEYS:
                out[k] = "<redacted>"
            else:
                out[k] = _scrub_secrets(v)
        return out
    if isinstance(obj, list):
        return [_scrub_secrets(v) for v in obj]
    return obj


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def maybe_init_wandb_run(
    benchmark_config: BenchmarkConfig,
    *,
    run_kind: str = "benchmark",
    extra_tags: Optional[Iterable[str]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Initialize a wandb run if enabled in the benchmark configuration.

    Args:
        benchmark_config: The fully resolved benchmark configuration. The caller
            is expected to have resolved `benchmark_config.output_dir` already.
        run_kind: A short label describing the run (e.g. "benchmark", "sweep_summary").
        extra_tags: Optional tags appended to `wandb.tags`.
        extra_config: Optional extra config merged into wandb config.
    """
    if (
        not getattr(benchmark_config, "wandb", None)
        or not benchmark_config.wandb.enabled
    ):
        return

    try:
        from typing import Any as _Any

        import wandb  # type: ignore[import-not-found]

        wandb = cast(_Any, wandb)
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "Weights & Biases logging is enabled, but `wandb` could not be imported. "
            "Install wandb (e.g. `pip install wandb`) or disable `wandb.enabled`."
        ) from exc

    if getattr(wandb, "run", None):
        # Defensive: if a run is already active (e.g. user code), don't clobber it.
        logger.info("wandb.run already initialized; skipping wandb.init()")
        return

    cfg_dict = dataclass_to_dict(benchmark_config)
    assert isinstance(cfg_dict, dict), f"Expected dict, got {type(cfg_dict)}"
    cfg_dict = cast(Dict[str, Any], _scrub_secrets(cfg_dict))

    wandb_cfg: Dict[str, Any] = {"run_kind": run_kind, **cfg_dict}
    if extra_config:
        wandb_cfg.update(extra_config)

    out_dir = benchmark_config.output_dir
    run_name = benchmark_config.wandb.run_name or os.path.basename(out_dir.rstrip("/"))
    tags = list(benchmark_config.wandb.tags or [])
    if extra_tags:
        tags.extend([t for t in extra_tags if t])
    # Deduplicate while preserving order
    seen = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))]

    logger.info(
        "Initializing wandb run (project=%s group=%s name=%s dir=%s)",
        benchmark_config.wandb.project,
        benchmark_config.wandb.group,
        run_name,
        out_dir,
    )

    wandb.init(
        project=benchmark_config.wandb.project,
        entity=benchmark_config.wandb.entity,
        group=benchmark_config.wandb.group,
        name=run_name,
        tags=tags or None,
        notes=benchmark_config.wandb.notes,
        dir=out_dir,
        config=wandb_cfg,
        reinit="finish_previous",
        mode=benchmark_config.wandb.mode,
    )


def maybe_log_sweep_summary(
    *, sweep_dir: str, wandb_cfg: WandbConfig, group: Optional[str] = None
) -> None:
    """Create a small wandb summary run for a CLI sweep (if enabled)."""
    if not wandb_cfg.enabled:
        return

    try:
        from typing import Any as _Any

        import wandb  # type: ignore[import-not-found]

        wandb = cast(_Any, wandb)
    except Exception:
        return

    name = f"sweep-summary-{os.path.basename(sweep_dir.rstrip('/'))}"
    wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        group=group or wandb_cfg.group,
        name=name,
        tags=dedup_tags([*wandb_cfg.tags, "sweep", "summary"]) or None,
        dir=sweep_dir,
        config={"sweep_dir": sweep_dir},
        reinit="finish_previous",
        mode=wandb_cfg.mode,
    )
    try:
        csv_path = os.path.join(sweep_dir, "sweep_summary.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = DictReader(f)
                rows = list(reader)
                if reader.fieldnames and rows:
                    cols = list(reader.fieldnames)
                    data = [[r.get(c) for c in cols] for r in rows]
                    table = wandb.Table(columns=cols, data=data)
                    wandb.log({"sweep_summary_table": table})

        artifact = wandb.Artifact(
            name=f"sweep-summary-files-{wandb.run.id}",
            type="veeksha-sweep",
        )
        has_entries = False
        for rel in (
            "sweep_manifest.json",
            "sweep_summary.json",
            "sweep_summary.csv",
            "decode_stats.json",
            "decode_stats.csv",
        ):
            p = os.path.join(sweep_dir, rel)
            if os.path.exists(p):
                artifact.add_file(p, name=rel)
                has_entries = True
        if has_entries:
            wandb.log_artifact(artifact)
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


def maybe_log_capacity_search_summary(
    *,
    output_dir: str,
    wandb_cfg: WandbConfig,
    result: Dict[str, Any],
    group: Optional[str] = None,
) -> None:
    """Create a wandb summary run for a capacity search (if enabled)."""
    if not wandb_cfg.enabled:
        return

    try:
        from typing import Any as _Any

        import wandb  # type: ignore[import-not-found]

        wandb = cast(_Any, wandb)
    except Exception:
        return

    name = f"capsearch-summary-{os.path.basename(output_dir.rstrip('/'))}"
    wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        group=group or wandb_cfg.group,
        name=name,
        tags=dedup_tags([*wandb_cfg.tags, "capsearch", "summary"]) or None,
        dir=output_dir,
        config={
            "capacity_search_output_dir": output_dir,
            "searched_knob": result.get("searched_knob"),
            "traffic_scheduler_type": result.get("traffic_scheduler_type"),
            "start_value": result.get("start_value"),
            "max_value": result.get("max_value"),
            "expansion_factor": result.get("expansion_factor"),
            "max_iterations": result.get("max_iterations"),
            "precision": result.get("precision"),
            "best_value": result.get("best_value"),
        },
        reinit="finish_previous",
        mode=wandb_cfg.mode,
    )

    try:
        history = result.get("history")
        if isinstance(history, list) and history:
            # Build both:
            # - a wide per-attempt table with per-SLO columns (observed/threshold/met)
            # - a long table with one row per (attempt, SLO)

            def _slo_key(slo: dict[str, Any], idx: int) -> str:
                raw = (
                    slo.get("name")
                    or slo.get("slo_metric_key")
                    or slo.get("metric")
                    or f"slo_{idx}"
                )
                s = str(raw)
                # make it column-friendly
                for ch in ["/", "\\", "\n", "\t"]:
                    s = s.replace(ch, " ")
                s = " ".join(s.split())
                return s

            # Discover all SLO keys across attempts to build stable columns.
            all_slo_keys: list[str] = []
            seen_keys: set[str] = set()
            for entry in history:
                if not isinstance(entry, dict):
                    continue
                slo_results = entry.get("slo_results")
                if not isinstance(slo_results, dict):
                    continue
                slos = slo_results.get("results")
                if not isinstance(slos, list):
                    continue
                for i, slo in enumerate(slos):
                    if not isinstance(slo, dict):
                        continue
                    key = _slo_key(slo, i)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    all_slo_keys.append(key)

            attempt_rows: list[dict[str, Any]] = []
            slo_detail_rows: list[dict[str, Any]] = []

            for idx, entry in enumerate(history, start=1):
                if not isinstance(entry, dict):
                    continue

                base_row: dict[str, Any] = {
                    "attempt": idx,
                    "value": entry.get("value"),
                    "all_slos_met": entry.get("all_slos_met"),
                    "run_dir": entry.get("run_dir"),
                }

                slo_results = entry.get("slo_results")
                if isinstance(slo_results, dict) and isinstance(
                    slo_results.get("results"), list
                ):
                    slos = cast(list[Any], slo_results.get("results"))
                    for i, slo in enumerate(slos):
                        if not isinstance(slo, dict):
                            continue
                        key = _slo_key(slo, i)
                        base_row[f"{key}/observed_value"] = slo.get("observed_value")
                        base_row[f"{key}/threshold"] = slo.get("threshold")
                        base_row[f"{key}/met"] = slo.get("met")

                        slo_detail_rows.append(
                            {
                                "attempt": idx,
                                "value": entry.get("value"),
                                "run_dir": entry.get("run_dir"),
                                "slo_key": key,
                                "met": slo.get("met"),
                                "observed_value": slo.get("observed_value"),
                                "threshold": slo.get("threshold"),
                                "percentile": slo.get("percentile"),
                                "metric": slo.get("metric"),
                                "slo_metric_key": slo.get("slo_metric_key"),
                                "lower_is_better": slo.get("lower_is_better"),
                            }
                        )

                # Ensure all SLO columns exist for table consistency.
                for key in all_slo_keys:
                    base_row.setdefault(f"{key}/observed_value", None)
                    base_row.setdefault(f"{key}/threshold", None)
                    base_row.setdefault(f"{key}/met", None)

                attempt_rows.append(base_row)

            if attempt_rows:
                # Keep the first columns stable, then per-SLO columns grouped.
                cols = ["attempt", "value", "all_slos_met", "run_dir"]
                for key in all_slo_keys:
                    cols.extend(
                        [
                            f"{key}/observed_value",
                            f"{key}/threshold",
                            f"{key}/met",
                        ]
                    )
                data = [[r.get(c) for c in cols] for r in attempt_rows]
                wandb.log({"capsearch_attempts": wandb.Table(columns=cols, data=data)})

                csv_path = os.path.join(output_dir, "capacity_search_attempts.csv")
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = DictWriter(f, fieldnames=cols)
                    writer.writeheader()
                    writer.writerows(attempt_rows)

            if slo_detail_rows:
                cols = [
                    "attempt",
                    "value",
                    "run_dir",
                    "slo_key",
                    "met",
                    "observed_value",
                    "threshold",
                    "percentile",
                    "metric",
                    "slo_metric_key",
                    "lower_is_better",
                ]
                data = [[r.get(c) for c in cols] for r in slo_detail_rows]
                wandb.log(
                    {"capsearch_slo_details": wandb.Table(columns=cols, data=data)}
                )

                csv_path = os.path.join(output_dir, "capacity_search_slo_details.csv")
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = DictWriter(f, fieldnames=cols)
                    writer.writeheader()
                    writer.writerows(slo_detail_rows)

        artifact = wandb.Artifact(
            name=f"capsearch-output-files-{wandb.run.id}",
            type="veeksha-capsearch",
        )
        has_entries = False
        for rel in (
            "config.yml",
            "capacity_search_results.json",
            "capacity_search_attempts.csv",
            "capacity_search_slo_details.csv",
        ):
            p = os.path.join(output_dir, rel)
            if os.path.exists(p):
                artifact.add_file(p, name=rel)
                has_entries = True
        if has_entries:
            wandb.log_artifact(artifact)
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


def maybe_persist_wandb_run_info(output_dir: str) -> None:
    """Persist basic wandb run identifiers in `output_dir/wandb_run.json`."""
    try:
        from typing import Any as _Any

        import wandb  # type: ignore[import-not-found]

        wandb = cast(_Any, wandb)
        if not getattr(wandb, "run", None):
            return
        run_info = {
            "id": getattr(wandb.run, "id", None),
            "name": getattr(wandb.run, "name", None),
            "entity": getattr(wandb.run, "entity", None),
            "project": getattr(wandb.run, "project", None),
            "group": getattr(wandb.run, "group", None),
            "path": getattr(wandb.run, "path", None),
            "url": getattr(wandb.run, "url", None),
        }
        with open(
            os.path.join(output_dir, "wandb_run.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(run_info, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to persist wandb run info: %s", exc)


def maybe_log_benchmark_scalars(output_dir: str) -> None:
    """Log common scalar metrics (summary, throughput, SLO) to wandb."""
    try:
        from typing import Any as _Any

        import wandb  # type: ignore[import-not-found]

        wandb = cast(_Any, wandb)
        if not getattr(wandb, "run", None):
            return
    except Exception:
        return

    summary = _safe_read_json(os.path.join(output_dir, "metrics", "summary_stats.json"))
    if summary:
        flat: Dict[str, float] = {}
        for k, v in summary.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[f"summary/{k}"] = float(v)
        if flat:
            wandb.log(flat)

    throughput = _safe_read_json(
        os.path.join(output_dir, "metrics", "throughput_metrics.json")
    )
    if throughput:
        flat = {}
        for k, v in throughput.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[f"throughput/{k}"] = float(v)
        if flat:
            wandb.log(flat)

    slo = _safe_read_json(os.path.join(output_dir, "metrics", "slo_results.json"))
    if slo:
        all_met = slo.get("all_slos_met")
        if isinstance(all_met, bool):
            wandb.log({"slo/all_slos_met": float(all_met)})


def _iter_artifact_files(output_dir: str) -> list[tuple[str, str]]:
    """Return a list of (absolute_path, artifact_name) files to upload."""
    root = Path(output_dir)
    if not root.exists():
        return []

    include_exts = {".yml", ".yaml", ".json", ".jsonl", ".csv", ".png", ".txt", ".log"}
    excluded_dir_names = {"wandb", ".wandb", "__pycache__"}

    selected: list[tuple[str, str]] = []

    # Always include run root config/health when present.
    for rel in ("config.yml", "health_check_results.txt", "wandb_run.json"):
        p = root / rel
        if p.exists() and p.is_file():
            selected.append((str(p), rel))

    # Include metrics + traces artifacts.
    for subdir in ("metrics", "traces"):
        base = root / subdir
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(part in excluded_dir_names for part in p.parts):
                continue
            if p.suffix.lower() not in include_exts:
                continue
            rel_name = str(p.relative_to(root))
            selected.append((str(p), rel_name))

    # Deterministic order.
    selected.sort(key=lambda x: x[1])
    return selected


def maybe_log_benchmark_artifacts(
    benchmark_config: BenchmarkConfig,
    *,
    artifact_name: Optional[str] = None,
    artifact_type: str = "veeksha-benchmark",
) -> None:
    """Upload selected output files as a wandb artifact."""
    if not benchmark_config.wandb.enabled or not benchmark_config.wandb.log_artifacts:
        return

    try:
        from typing import Any as _Any

        import wandb  # type: ignore[import-not-found]

        wandb = cast(_Any, wandb)
        if not getattr(wandb, "run", None):
            return
    except Exception:
        return

    files = _iter_artifact_files(benchmark_config.output_dir)
    if not files:
        return

    name = artifact_name or f"benchmark-output-files-{wandb.run.id}"
    artifact = wandb.Artifact(name=name, type=artifact_type)
    has_entries = False
    for abs_path, rel_name in files:
        if os.path.exists(abs_path):
            artifact.add_file(abs_path, name=rel_name)
            has_entries = True

    if has_entries:
        wandb.log_artifact(artifact)


def maybe_finish_wandb_run(output_dir: str) -> None:
    """Finish the active wandb run if one exists."""
    try:
        from typing import Any as _Any

        import wandb  # type: ignore[import-not-found]

        wandb = cast(_Any, wandb)
        if not getattr(wandb, "run", None):
            return
        maybe_persist_wandb_run_info(output_dir)
        try:
            wandb.finish()
        except Exception as exc:
            logger.warning("wandb.finish() failed: %s", exc)
    except Exception:
        return


def update_run_tags(output_dir: str, tags: Iterable[str]) -> None:
    """Update tags for the wandb run associated with the given output directory."""
    try:
        from typing import Any as _Any

        import wandb  # type: ignore[import-not-found]

        wandb = cast(_Any, wandb)
    except Exception:
        return

    run_info = _safe_read_json(os.path.join(output_dir, "wandb_run.json"))
    if not run_info:
        return

    run_path = run_info.get("path")
    if not run_path:
        return

    try:
        api = wandb.Api()
        run = api.run(run_path)
        run.tags = dedup_tags([*run.tags, *tags])
        run.update()
        logger.info("Updated tags for run %s: %s", run_path, tags)
    except Exception as exc:
        logger.warning("Failed to update tags for run %s: %s", run_path, exc)


# ---------------------------------------------------------------------------
# Microbenchmark wandb helpers
# ---------------------------------------------------------------------------


def maybe_init_microbench_wandb_run(
    wandb_cfg: "WandbConfig",
    *,
    microbench_type: str,
    output_dir: str,
    config_dict: Dict[str, Any],
    extra_tags: Optional[Iterable[str]] = None,
) -> None:
    """Initialize a wandb run for a microbenchmark (prefill/decode/stress).

    Safe no-op when wandb is disabled or unavailable.
    """
    if not wandb_cfg.enabled:
        return

    try:
        from typing import Any as _Any

        import wandb  # type: ignore[import-not-found]

        wandb = cast(_Any, wandb)
    except Exception as exc:
        raise ImportError(
            "Weights & Biases logging is enabled, but `wandb` could not be imported. "
            "Install wandb (e.g. `pip install wandb`) or disable `wandb.enabled`."
        ) from exc

    if getattr(wandb, "run", None):
        logger.info("wandb.run already initialized; skipping wandb.init()")
        return

    config_dict = cast(Dict[str, Any], _scrub_secrets(config_dict))
    wandb_cfg_dict: Dict[str, Any] = {
        "run_kind": f"microbench-{microbench_type}",
        **config_dict,
    }

    run_name = wandb_cfg.run_name or os.path.basename(output_dir.rstrip("/"))
    tags = dedup_tags(
        [*wandb_cfg.tags, "microbench", microbench_type, *(extra_tags or [])]
    )

    logger.info(
        "Initializing wandb run for microbench-%s (project=%s name=%s)",
        microbench_type,
        wandb_cfg.project,
        run_name,
    )

    wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        group=wandb_cfg.group,
        name=run_name,
        tags=tags or None,
        notes=wandb_cfg.notes,
        dir=output_dir,
        config=wandb_cfg_dict,
        reinit="finish_previous",
        mode=wandb_cfg.mode,
    )


def _log_microbench_prefill(results: Dict[str, Any], wandb: Any) -> None:
    """Log prefill microbenchmark results to wandb."""
    rows = results.get("results", [])
    if not rows:
        return

    rows = sorted(rows, key=lambda r: r["input_length"])

    # Scalars per input length
    for row in rows:
        il = row["input_length"]
        ttfc = row.get("ttfc", {})
        wandb.log(
            {
                "prefill/input_length": il,
                "prefill/ttfc_p50_ms": ttfc.get("median", 0) * 1000,
                "prefill/ttfc_p99_ms": ttfc.get("p99", 0) * 1000,
                "prefill/ttfc_mean_ms": ttfc.get("mean", 0) * 1000,
            }
        )

    # Table
    columns = [
        "input_length",
        "mean_ms",
        "p50_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "count",
    ]
    data = []
    for row in rows:
        s = row.get("ttfc", {})
        data.append(
            [
                row["input_length"],
                round(s.get("mean", 0) * 1000, 2),
                round(s.get("median", 0) * 1000, 2),
                round(s.get("p99", 0) * 1000, 2),
                round(s.get("min", 0) * 1000, 2),
                round(s.get("max", 0) * 1000, 2),
                s.get("count", 0),
            ]
        )
    wandb.log({"prefill_results": wandb.Table(columns=columns, data=data)})

    # Line charts
    xs = [r["input_length"] for r in rows]
    wandb.log(
        {
            "prefill/ttfc_vs_input_length": wandb.plot.line_series(
                xs=xs,
                ys=[
                    [r.get("ttfc", {}).get("median", 0) * 1000 for r in rows],
                    [r.get("ttfc", {}).get("p99", 0) * 1000 for r in rows],
                ],
                keys=["P50", "P99"],
                title="TTFC vs Input Length",
                xname="Input Length (tokens)",
            ),
        }
    )


def _log_microbench_decode(results: Dict[str, Any], wandb: Any) -> None:
    """Log decode microbenchmark results to wandb."""
    rows = results.get("results", [])
    if not rows:
        return

    rows = sorted(rows, key=lambda r: (r["batch_size"], r["input_length"]))

    # Scalars per (batch_size, input_length)
    for row in rows:
        bs = row["batch_size"]
        il = row["input_length"]
        tbt = row.get("tbt", {})
        wandb.log(
            {
                "decode/batch_size": bs,
                "decode/input_length": il,
                "decode/tbt_p50_ms": tbt.get("median", 0) * 1000,
                "decode/tbt_p99_ms": tbt.get("p99", 0) * 1000,
                "decode/tbt_mean_ms": tbt.get("mean", 0) * 1000,
            }
        )

    # Table
    columns = [
        "batch_size",
        "input_length",
        "mean_ms",
        "p50_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "samples",
    ]
    data = []
    for row in rows:
        s = row.get("tbt", {})
        data.append(
            [
                row["batch_size"],
                row["input_length"],
                round(s.get("mean", 0) * 1000, 2),
                round(s.get("median", 0) * 1000, 2),
                round(s.get("p99", 0) * 1000, 2),
                round(s.get("min", 0) * 1000, 2),
                round(s.get("max", 0) * 1000, 2),
                s.get("count", 0),
            ]
        )
    wandb.log({"decode_results": wandb.Table(columns=columns, data=data)})

    # Line charts: TBT vs batch size, one series per input_length
    from collections import defaultdict

    by_il: Dict[int, list] = defaultdict(list)
    for r in rows:
        by_il[r["input_length"]].append(r)

    if len(by_il) >= 1:
        # Group by input_length → TBT P50 vs batch_size
        all_bs = sorted({r["batch_size"] for r in rows})
        series_keys = []
        series_ys = []
        for il in sorted(by_il):
            bs_map = {
                r["batch_size"]: r.get("tbt", {}).get("median", 0) * 1000
                for r in by_il[il]
            }
            series_keys.append(f"IL={il}")
            series_ys.append([bs_map.get(bs, None) for bs in all_bs])
        wandb.log(
            {
                "decode/tbt_p50_vs_batch_size": wandb.plot.line_series(
                    xs=all_bs,
                    ys=series_ys,
                    keys=series_keys,
                    title="TBT P50 vs Batch Size",
                    xname="Batch Size",
                ),
            }
        )

    # Group by batch_size → TBT P50 vs input_length
    by_bs: Dict[int, list] = defaultdict(list)
    for r in rows:
        by_bs[r["batch_size"]].append(r)

    if len(by_bs) >= 1:
        all_il = sorted({r["input_length"] for r in rows})
        series_keys = []
        series_ys = []
        for bs in sorted(by_bs):
            il_map = {
                r["input_length"]: r.get("tbt", {}).get("median", 0) * 1000
                for r in by_bs[bs]
            }
            series_keys.append(f"BS={bs}")
            series_ys.append([il_map.get(il, None) for il in all_il])
        wandb.log(
            {
                "decode/tbt_p50_vs_input_length": wandb.plot.line_series(
                    xs=all_il,
                    ys=series_ys,
                    keys=series_keys,
                    title="TBT P50 vs Input Length",
                    xname="Input Length (tokens)",
                ),
            }
        )


def _log_microbench_stress(results: Dict[str, Any], wandb: Any) -> None:
    """Log stress microbenchmark results to wandb."""
    rows = results.get("results", [])
    if not rows:
        return

    num_gpus = results.get("num_gpus", 0)

    # Step-indexed scalars (one wandb.log per concurrency level)
    for r in rows:
        metrics: Dict[str, Any] = {
            "stress/level": r["level"],
            "stress/output_throughput": r["output_throughput"],
            "stress/input_throughput": r["input_throughput"],
            "stress/e2e_latency_p50_ms": r["e2e_latency_p50"] * 1000,
            "stress/e2e_latency_p99_ms": r["e2e_latency_p99"] * 1000,
            "stress/ttfc_p50_ms": r["ttfc_p50"] * 1000,
            "stress/ttfc_p99_ms": r["ttfc_p99"] * 1000,
            "stress/interactivity_p50": r["interactivity_p50"],
            "stress/interactivity_p99": r["interactivity_p99"],
            "stress/num_requests": r["num_requests"],
        }
        if num_gpus > 0:
            metrics["stress/output_tps_per_gpu"] = r.get(
                "output_tps_per_gpu", r["output_throughput"] / num_gpus
            )
            metrics["stress/input_tps_per_gpu"] = r.get(
                "input_tps_per_gpu", r["input_throughput"] / num_gpus
            )
        wandb.log(metrics, step=r["level"])

    # Table
    columns = [
        "level",
        "output_throughput",
        "input_throughput",
        "e2e_p50_ms",
        "e2e_p99_ms",
        "ttfc_p50_ms",
        "ttfc_p99_ms",
        "interactivity_p50",
        "interactivity_p99",
        "num_requests",
    ]
    if num_gpus > 0:
        columns.extend(["output_tps_per_gpu", "input_tps_per_gpu"])

    data = []
    for r in rows:
        row = [
            r["level"],
            round(r["output_throughput"], 1),
            round(r["input_throughput"], 1),
            round(r["e2e_latency_p50"] * 1000, 1),
            round(r["e2e_latency_p99"] * 1000, 1),
            round(r["ttfc_p50"] * 1000, 1),
            round(r["ttfc_p99"] * 1000, 1),
            round(r["interactivity_p50"], 1),
            round(r["interactivity_p99"], 1),
            r["num_requests"],
        ]
        if num_gpus > 0:
            row.extend(
                [
                    round(
                        r.get("output_tps_per_gpu", r["output_throughput"] / num_gpus),
                        1,
                    ),
                    round(
                        r.get("input_tps_per_gpu", r["input_throughput"] / num_gpus), 1
                    ),
                ]
            )
        data.append(row)
    wandb.log({"stress_results": wandb.Table(columns=columns, data=data)})

    # Line charts
    levels = [r["level"] for r in rows]

    wandb.log(
        {
            "stress/throughput_vs_load": wandb.plot.line_series(
                xs=levels,
                ys=[
                    [r["output_throughput"] for r in rows],
                    [r["input_throughput"] for r in rows],
                ],
                keys=["Output (tok/s)", "Input (tok/s)"],
                title="Throughput vs Load",
                xname="Concurrency",
            ),
        }
    )

    wandb.log(
        {
            "stress/latency_vs_load": wandb.plot.line_series(
                xs=levels,
                ys=[
                    [r["e2e_latency_p50"] * 1000 for r in rows],
                    [r["e2e_latency_p99"] * 1000 for r in rows],
                    [r["ttfc_p50"] * 1000 for r in rows],
                    [r["ttfc_p99"] * 1000 for r in rows],
                ],
                keys=["E2E P50", "E2E P99", "TTFC P50", "TTFC P99"],
                title="Latency vs Load (ms)",
                xname="Concurrency",
            ),
        }
    )

    wandb.log(
        {
            "stress/interactivity_vs_load": wandb.plot.line_series(
                xs=levels,
                ys=[
                    [r["interactivity_p50"] for r in rows],
                    [r["interactivity_p99"] for r in rows],
                ],
                keys=["P50 (tok/s/user)", "P99 (tok/s/user)"],
                title="Interactivity vs Load",
                xname="Concurrency",
            ),
        }
    )

    if num_gpus > 0:
        tps_per_gpu = [
            r.get("output_tps_per_gpu", r["output_throughput"] / num_gpus) for r in rows
        ]
        wandb.log(
            {
                "stress/tps_per_gpu_vs_load": wandb.plot.line_series(
                    xs=levels,
                    ys=[
                        tps_per_gpu,
                        [
                            r.get("input_tps_per_gpu", r["input_throughput"] / num_gpus)
                            for r in rows
                        ],
                    ],
                    keys=["Output TPS/GPU", "Input TPS/GPU"],
                    title="TPS/GPU vs Load",
                    xname="Concurrency",
                ),
            }
        )

        wandb.log(
            {
                "stress/throughput_curve": wandb.plot.line_series(
                    xs=[r["interactivity_p50"] for r in rows],
                    ys=[tps_per_gpu],
                    keys=["TPS/GPU"],
                    title="Throughput Curve: TPS/GPU vs TPS/User",
                    xname="TPS/User (tok/s/user)",
                ),
            }
        )

    # Summary
    if rows:
        peak = max(rows, key=lambda r: r["output_throughput"])
        wandb.run.summary["peak_output_throughput"] = peak["output_throughput"]
        wandb.run.summary["interactivity_at_peak"] = peak["interactivity_p50"]
        wandb.run.summary["peak_concurrency"] = peak["level"]
        if num_gpus > 0:
            wandb.run.summary["peak_tps_per_gpu"] = peak["output_throughput"] / num_gpus


_MICROBENCH_LOGGERS = {
    "prefill": _log_microbench_prefill,
    "decode": _log_microbench_decode,
    "stress": _log_microbench_stress,
}


def maybe_log_microbench_results(
    wandb_cfg: "WandbConfig",
    *,
    microbench_type: str,
    results_json_path: str,
    output_dir: str,
) -> None:
    """Log microbenchmark results (tables, scalars, plots, artifacts) to wandb.

    Safe no-op when wandb is disabled or no active run.
    """
    if not wandb_cfg.enabled:
        return

    try:
        from typing import Any as _Any

        import wandb  # type: ignore[import-not-found]

        wandb = cast(_Any, wandb)
        if not getattr(wandb, "run", None):
            return
    except Exception:
        return

    # Log type-specific metrics
    if os.path.exists(results_json_path):
        with open(results_json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        log_fn = _MICROBENCH_LOGGERS.get(microbench_type)
        if log_fn:
            try:
                log_fn(results, wandb)
            except Exception as exc:
                logger.warning(
                    "Failed to log %s results to wandb: %s", microbench_type, exc
                )

    # Log plot images
    plots_dir = os.path.join(output_dir, "plots")
    if os.path.isdir(plots_dir):
        for fname in sorted(os.listdir(plots_dir)):
            if fname.endswith(".png"):
                try:
                    wandb.log({fname[:-4]: wandb.Image(os.path.join(plots_dir, fname))})
                except Exception:
                    pass

    # Upload artifacts
    if wandb_cfg.log_artifacts:
        artifact = wandb.Artifact(
            name=f"microbench-{microbench_type}-{wandb.run.id}",
            type=f"veeksha-microbench-{microbench_type}",
        )
        has_entries = False
        for fname in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, fname)
            if os.path.isfile(fpath) and any(
                fname.endswith(ext) for ext in (".json", ".csv")
            ):
                artifact.add_file(fpath, name=fname)
                has_entries = True
        if os.path.isdir(plots_dir):
            for fname in sorted(os.listdir(plots_dir)):
                fpath = os.path.join(plots_dir, fname)
                if os.path.isfile(fpath):
                    artifact.add_file(fpath, name=f"plots/{fname}")
                    has_entries = True
        if has_entries:
            wandb.log_artifact(artifact)


def maybe_finish_microbench_wandb_run(
    wandb_cfg: "WandbConfig",
    output_dir: str,
) -> None:
    """Finish a microbenchmark wandb run.

    Safe no-op when wandb is disabled or no active run.
    """
    if not wandb_cfg.enabled:
        return

    try:
        from typing import Any as _Any

        import wandb  # type: ignore[import-not-found]

        wandb = cast(_Any, wandb)
        if not getattr(wandb, "run", None):
            return
        maybe_persist_wandb_run_info(output_dir)
        try:
            wandb.finish()
        except Exception as exc:
            logger.warning("wandb.finish() failed: %s", exc)
    except Exception:
        return
