"""Shared utilities for ad hoc benchmark plotting/report scripts.

This module contains reusable run discovery/selection helpers and selected-run
archive helpers for sweep plotting scripts.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NARADA_DIR = REPO_ROOT / "narada"
DATA_SUFFIXES = {".csv", ".json", ".jsonl", ".txt", ".yaml", ".yml"}
ROOT_DATA_FILES = ("config.yml", "health_check_results.txt", "wandb_run.json")

PERCENTILES: Tuple[str, ...] = ("P50", "P90")
REPORT_METRICS: Tuple[Tuple[str, str, str, bool], ...] = (
    ("ttfa", "TTFA", "ms", True),
    ("rtf", "RTF", "ratio", True),
    ("generated_audio_duration", "Generated Audio Duration", "ms", False),
)

# DD_MM_YYYY-HH_MM_SS-<hash>
DIR_TS_RE = re.compile(r"^(\d{2}_\d{2}_\d{4}-\d{2}_\d{2}_\d{2})-")


@dataclass(frozen=True)
class Run:
    system: str
    run_dir: str
    concurrency: int
    timestamp: datetime
    completed: Optional[int]
    summary: Dict[str, Any]


@dataclass(frozen=True)
class InputRun:
    system: str
    run_dir: str
    input_chars: int
    concurrency: Optional[int]
    timestamp: datetime
    completed: Optional[int]
    summary: Dict[str, Any]


SweepRun = Run | InputRun


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_read_json(path: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def safe_read_yaml(path: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def parse_run_timestamp(dirname: str) -> Optional[datetime]:
    match = DIR_TS_RE.match(dirname)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%d_%m_%Y-%H_%M_%S")
    except ValueError:
        return None


def find_key(cfg: Dict[str, Any], target: str) -> Optional[Any]:
    stack: List[Any] = [cfg]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if target in node:
                return node[target]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


def find_concurrency(cfg: Dict[str, Any]) -> Optional[int]:
    value = find_key(cfg, "target_concurrent_sessions")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def required_metric_keys(
    metrics: Sequence[Tuple[str, str, str, bool]] = REPORT_METRICS,
    percentiles: Sequence[str] = PERCENTILES,
) -> List[str]:
    return [f"{metric[0]} ({p})" for metric in metrics for p in percentiles]


def collect_runs(
    system: str,
    root: str | Path,
    min_completed: int,
    metrics: Sequence[Tuple[str, str, str, bool]] = REPORT_METRICS,
    percentiles: Sequence[str] = PERCENTILES,
) -> List[Run]:
    runs: List[Run] = []
    root = Path(root)
    if not root.is_dir():
        print(f"[warn] {system}: directory does not exist: {root}", file=sys.stderr)
        return runs

    required_keys = required_metric_keys(metrics, percentiles)
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        ts = parse_run_timestamp(path.name)
        if ts is None:
            continue
        cfg = safe_read_yaml(path / "config.yml")
        if cfg is None:
            continue
        concurrency = find_concurrency(cfg)
        if concurrency is None:
            continue
        summary = safe_read_json(path / "metrics" / "summary_stats.json")
        if summary is None or any(k not in summary for k in required_keys):
            continue
        completed = as_int(summary.get("Number of Completed Requests"))
        if min_completed > 0 and (completed is None or completed < min_completed):
            continue
        runs.append(Run(system, str(path), concurrency, ts, completed, summary))
    return runs


def collect_input_runs(
    system: str,
    root: str | Path,
    min_completed: int,
    metrics: Sequence[Tuple[str, str, str, bool]] = REPORT_METRICS,
    percentiles: Sequence[str] = PERCENTILES,
) -> List[InputRun]:
    runs: List[InputRun] = []
    root = Path(root)
    if not root.is_dir():
        print(f"[warn] {system}: directory does not exist: {root}", file=sys.stderr)
        return runs

    required_keys = required_metric_keys(metrics, percentiles)
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        ts = parse_run_timestamp(path.name)
        if ts is None:
            continue
        cfg = safe_read_yaml(path / "config.yml")
        if cfg is None:
            continue
        input_chars = as_int(find_key(cfg, "min_chars"))
        if input_chars is None or input_chars < 0:
            continue
        concurrency = as_int(find_key(cfg, "target_concurrent_sessions"))
        summary = safe_read_json(path / "metrics" / "summary_stats.json")
        if summary is None or any(k not in summary for k in required_keys):
            continue
        completed = as_int(summary.get("Number of Completed Requests"))
        if min_completed > 0 and (completed is None or completed < min_completed):
            continue
        runs.append(
            InputRun(
                system,
                str(path),
                input_chars,
                concurrency,
                ts,
                completed,
                summary,
            )
        )
    return runs


def run_rtf(run: Run | InputRun) -> float:
    try:
        rtf = float(run.summary["rtf (P90)"])
    except (KeyError, TypeError, ValueError):
        return float("inf")
    return rtf if rtf > 0 else float("inf")


def pick_best_rtf(runs: Sequence[Run]) -> Dict[Tuple[str, int], Run]:
    best: Dict[Tuple[str, int], Run] = {}
    for run in runs:
        key = (run.system, run.concurrency)
        current = best.get(key)
        if current is None:
            best[key] = run
            continue
        current_rtf, new_rtf = run_rtf(current), run_rtf(run)
        if new_rtf < current_rtf or (
            new_rtf == current_rtf and run.timestamp > current.timestamp
        ):
            best[key] = run
    return best


def pick_best_input_rtf(
    runs: Sequence[InputRun],
) -> Dict[Tuple[str, int], InputRun]:
    best: Dict[Tuple[str, int], InputRun] = {}
    for run in runs:
        key = (run.system, run.input_chars)
        current = best.get(key)
        if current is None:
            best[key] = run
            continue
        current_rtf, new_rtf = run_rtf(current), run_rtf(run)
        if new_rtf < current_rtf or (
            new_rtf == current_rtf and run.timestamp > current.timestamp
        ):
            best[key] = run
    return best


def _metric_rows_for_run(
    *,
    system: str,
    axis_name: str,
    axis_value: int,
    run_dir: str,
    completed: Optional[int],
    summary: Dict[str, Any],
    metrics: Sequence[Tuple[str, str, str, bool]],
    percentiles: Sequence[str],
    concurrency: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, display, _, _ in metrics:
        for percentile in percentiles:
            row: Dict[str, Any] = {
                "system": system,
                axis_name: axis_value,
                "metric": display,
                "percentile": percentile,
                "value": safe_float(summary.get(f"{key} ({percentile})")),
                "completed_requests": completed,
                "chars_per_sec_aggregate": safe_float(
                    summary.get("chars_per_sec_aggregate")
                ),
                "run_dir": Path(run_dir).name,
            }
            if concurrency is not None:
                row["concurrency"] = concurrency
            rows.append(row)
    return rows


def build_concurrency_df(
    selected: Dict[Tuple[str, int], Run],
    systems: Sequence[str],
    metrics: Sequence[Tuple[str, str, str, bool]] = REPORT_METRICS,
    percentiles: Sequence[str] = PERCENTILES,
):
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for (system, concurrency), run in selected.items():
        rows.extend(
            _metric_rows_for_run(
                system=system,
                axis_name="concurrency",
                axis_value=concurrency,
                run_dir=run.run_dir,
                completed=run.completed,
                summary=run.summary,
                metrics=metrics,
                percentiles=percentiles,
            )
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    all_axis = sorted({int(c) for c in df["concurrency"]})
    existing = {(s, int(c)) for s, c in zip(df["system"], df["concurrency"])}
    missing_rows: List[Dict[str, Any]] = []
    for system in systems:
        for axis_value in all_axis:
            if (system, axis_value) in existing:
                continue
            for _, display, _, _ in metrics:
                for percentile in percentiles:
                    missing_rows.append(
                        {
                            "system": system,
                            "concurrency": axis_value,
                            "metric": display,
                            "percentile": percentile,
                            "value": float("nan"),
                            "completed_requests": float("nan"),
                            "chars_per_sec_aggregate": float("nan"),
                            "run_dir": "",
                        }
                    )
    if missing_rows:
        df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

    df["concurrency"] = pd.Categorical(
        df["concurrency"].astype(str),
        categories=[str(c) for c in all_axis],
        ordered=True,
    )
    df["system"] = pd.Categorical(df["system"], categories=list(systems), ordered=True)
    df["percentile"] = pd.Categorical(
        df["percentile"], categories=list(percentiles), ordered=True
    )
    df["metric"] = pd.Categorical(
        df["metric"], categories=[metric[1] for metric in metrics], ordered=True
    )
    return df.sort_values(
        by=["metric", "percentile", "concurrency", "system"]
    ).reset_index(drop=True)


def build_input_df(
    selected: Dict[Tuple[str, int], InputRun],
    systems: Sequence[str],
    metrics: Sequence[Tuple[str, str, str, bool]] = REPORT_METRICS,
    percentiles: Sequence[str] = PERCENTILES,
):
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for (system, input_chars), run in selected.items():
        rows.extend(
            _metric_rows_for_run(
                system=system,
                axis_name="input_chars",
                axis_value=input_chars,
                run_dir=run.run_dir,
                completed=run.completed,
                summary=run.summary,
                metrics=metrics,
                percentiles=percentiles,
                concurrency=run.concurrency,
            )
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    all_axis = sorted({int(c) for c in df["input_chars"]})
    existing = {(s, int(c)) for s, c in zip(df["system"], df["input_chars"])}
    missing_rows: List[Dict[str, Any]] = []
    for system in systems:
        for axis_value in all_axis:
            if (system, axis_value) in existing:
                continue
            for _, display, _, _ in metrics:
                for percentile in percentiles:
                    missing_rows.append(
                        {
                            "system": system,
                            "input_chars": axis_value,
                            "concurrency": None,
                            "metric": display,
                            "percentile": percentile,
                            "value": float("nan"),
                            "completed_requests": float("nan"),
                            "chars_per_sec_aggregate": float("nan"),
                            "run_dir": "",
                        }
                    )
    if missing_rows:
        df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

    df["input_chars"] = pd.Categorical(
        df["input_chars"].astype(str),
        categories=[str(c) for c in all_axis],
        ordered=True,
    )
    df["system"] = pd.Categorical(df["system"], categories=list(systems), ordered=True)
    df["percentile"] = pd.Categorical(
        df["percentile"], categories=list(percentiles), ordered=True
    )
    df["metric"] = pd.Categorical(
        df["metric"], categories=[metric[1] for metric in metrics], ordered=True
    )
    return df.sort_values(
        by=["metric", "percentile", "input_chars", "system"]
    ).reset_index(drop=True)


def filter_complete_axis(df, axis_col: str, systems: Sequence[str]):
    if df.empty:
        return df
    presence = (
        df.groupby([axis_col, "system"], observed=True)["value"]
        .apply(lambda s: bool(s.notna().any()))
        .unstack("system")
    )
    keep = [
        axis_value
        for axis_value, row in presence.iterrows()
        if all(bool(row.get(system, False)) for system in systems)
    ]
    if not keep:
        return df.iloc[0:0]
    keep_str = [str(value) for value in keep]
    out = df[df[axis_col].astype(str).isin(keep_str)].copy()
    out[axis_col] = out[axis_col].astype(str)
    import pandas as pd

    out[axis_col] = pd.Categorical(out[axis_col], categories=keep_str, ordered=True)
    return out


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return slug.lower() or "unnamed"


def _run_created_at(run: SweepRun) -> str:
    return run.timestamp.strftime("%Y-%m-%dT%H:%M:%S")


def _run_date_for_dir(run: SweepRun) -> str:
    return run.timestamp.strftime("%Y_%m_%d-%H_%M_%S")


def _iter_data_files(run_dir: Path, include_plots: bool) -> Iterable[Path]:
    for name in ROOT_DATA_FILES:
        path = run_dir / name
        if path.is_file():
            yield path

    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        return

    for path in sorted(metrics_dir.rglob("*")):
        if not path.is_file():
            continue
        if include_plots or path.suffix.lower() in DATA_SUFFIXES:
            yield path


def _write_metadata(
    target_dir: Path,
    *,
    plot_name: str,
    engine_name: str,
    comparison_against: str,
    run: SweepRun,
    copied_files: Sequence[Path],
    selection_policy: Optional[str] = None,
) -> None:
    exported_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    source_run_dir = Path(run.run_dir).resolve()
    rel_files = [str(path.relative_to(target_dir)) for path in copied_files]

    lines = [
        f"plot_name: {plot_name}",
        f"engine_name: {engine_name}",
        f"comparison_against: {comparison_against}",
        f"data_created_at: {_run_created_at(run)}",
        f"exported_at_utc: {exported_at}",
        f"source_run_dir: {source_run_dir}",
    ]
    if run.concurrency is not None:
        lines.append(f"concurrency: {run.concurrency}")
    if isinstance(run, InputRun):
        lines.append(f"input_chars: {run.input_chars}")
    lines.append(
        "selection_policy: "
        + (
            selection_policy
            or "lowest positive RTF P90 per (engine, concurrency); "
            "latest timestamp tie-break"
        )
    )
    lines.extend(["", "copied_files:"])
    lines.extend(f"- {path}" for path in rel_files)
    lines.append("")
    (target_dir / "metadata.txt").write_text("\n".join(lines), encoding="utf-8")


def _copy_run_data(
    run: SweepRun,
    target_dir: Path,
    *,
    plot_name: str,
    comparison_against: str,
    include_plots: bool,
    overwrite: bool,
    dry_run: bool,
    selection_policy: Optional[str] = None,
) -> List[Path]:
    source_run_dir = Path(run.run_dir)
    files = list(_iter_data_files(source_run_dir, include_plots))
    if not files:
        raise RuntimeError(f"No data files found in {source_run_dir}")

    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target_dir} already exists; pass --overwrite to replace it"
            )
        if not dry_run:
            shutil.rmtree(target_dir)

    if dry_run:
        return [target_dir / path.relative_to(source_run_dir) for path in files]

    copied_files: List[Path] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        rel = src.relative_to(source_run_dir)
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_files.append(dst)

    _write_metadata(
        target_dir,
        plot_name=plot_name,
        engine_name=run.system,
        comparison_against=comparison_against,
        run=run,
        copied_files=copied_files,
        selection_policy=selection_policy,
    )
    return copied_files


def _sweep_archive_kind(sweep_type: str) -> str:
    return "conc" if sweep_type == "conc" else "input"


def _run_axis_value(run: SweepRun, axis_col: str) -> Optional[int]:
    if axis_col == "concurrency":
        return run.concurrency
    if isinstance(run, InputRun) and axis_col == "input_chars":
        return run.input_chars
    return None


def _axis_folder_name(run: SweepRun, *, axis_col: str, sweep_type: str) -> Optional[str]:
    axis_value = _run_axis_value(run, axis_col)
    if axis_value is None:
        return None
    return f"{_sweep_archive_kind(sweep_type)}={axis_value}"


def _selected_runs_for_axis_values(
    selected_runs: Sequence[SweepRun], axis_col: str, axis_values: Iterable[Any]
) -> List[SweepRun]:
    keep = {str(value) for value in axis_values}
    return [
        run
        for run in selected_runs
        if (axis_value := _run_axis_value(run, axis_col)) is not None
        and str(axis_value) in keep
    ]


def _model_name_from_run(run: SweepRun) -> Optional[str]:
    cfg = safe_read_yaml(Path(run.run_dir) / "config.yml")
    if cfg is None:
        return None
    model = find_key(cfg, "model")
    return str(model) if model else None


def _infer_model_name(selected_runs: Sequence[SweepRun]) -> str:
    model_names: List[str] = []
    for run in selected_runs:
        model_name = _model_name_from_run(run)
        if model_name and model_name not in model_names:
            model_names.append(model_name)

    if not model_names:
        print(
            "[warn] Could not infer a model name from selected runs; "
            "using unknown_model.",
            file=sys.stderr,
        )
        return "unknown_model"

    if len(model_names) > 1:
        print(
            f"[warn] Multiple model names found ({', '.join(model_names)}); "
            f"using {model_names[0]}. Pass --model-name to override.",
            file=sys.stderr,
        )
    return model_names[0]


def _selection_policy_for_sweep(sweep_type: str) -> str:
    axis = "concurrency" if sweep_type == "conc" else "input_chars"
    return f"lowest positive RTF P90 per (engine, {axis}); latest timestamp tie-break"


def _write_sweep_archive_manifest(
    sweep_dir: Path,
    rows: Sequence[Tuple[SweepRun, Path, str]],
    *,
    axis_col: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    manifest_path = sweep_dir / "selected_runs.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "engine_name",
                "comparison_against",
                axis_col,
                "concurrency",
                "input_chars",
                "data_created_at",
                "source_run_dir",
                "benchmark_data_dir",
            ]
        )
        for run, target_dir, comparison_against in rows:
            input_chars = run.input_chars if isinstance(run, InputRun) else ""
            axis_value = _run_axis_value(run, axis_col)
            writer.writerow(
                [
                    run.system,
                    comparison_against,
                    axis_value if axis_value is not None else "",
                    run.concurrency if run.concurrency is not None else "",
                    input_chars,
                    _run_created_at(run),
                    str(Path(run.run_dir).resolve()),
                    str(target_dir.resolve()),
                ]
            )


def archive_selected_sweep_runs(
    selected_runs: Sequence[SweepRun],
    *,
    axis_values: Iterable[Any],
    axis_col: str,
    sweep_type: str,
    systems: Sequence[str],
    exp_name: Optional[str],
    model_name: Optional[str],
    benchmarks_dir: str | Path,
    include_plots: bool,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if not exp_name:
        return

    archive_runs = _selected_runs_for_axis_values(selected_runs, axis_col, axis_values)
    if not archive_runs:
        print("[warn] No selected runs remain to archive.", file=sys.stderr)
        return

    resolved_model_name = model_name or _infer_model_name(archive_runs)
    sweep_dir = (
        Path(benchmarks_dir)
        / _slug(resolved_model_name)
        / _slug(exp_name)
        / "sweep"
        / _sweep_archive_kind(sweep_type)
    )
    comparison_for = {
        systems[0]: systems[1] if len(systems) > 1 else "none",
    }
    if len(systems) > 1:
        comparison_for[systems[1]] = systems[0]

    rows: List[Tuple[SweepRun, Path, str]] = []
    selection_policy = _selection_policy_for_sweep(sweep_type)
    for run in archive_runs:
        axis_folder = _axis_folder_name(run, axis_col=axis_col, sweep_type=sweep_type)
        if axis_folder is None:
            continue
        target_dir = sweep_dir / _slug(run.system) / axis_folder
        comparison_against = comparison_for.get(run.system, "unknown")
        copied_files = _copy_run_data(
            run,
            target_dir,
            plot_name=exp_name,
            comparison_against=comparison_against,
            include_plots=include_plots,
            overwrite=overwrite,
            dry_run=dry_run,
            selection_policy=selection_policy,
        )
        print(
            f"{'Would copy' if dry_run else 'Copied'} "
            f"{len(copied_files)} files -> {target_dir}"
        )
        rows.append((run, target_dir, comparison_against))

    _write_sweep_archive_manifest(
        sweep_dir,
        rows,
        axis_col=axis_col,
        dry_run=dry_run,
    )
    if not dry_run:
        print(f"Wrote benchmark archive under {sweep_dir}")
