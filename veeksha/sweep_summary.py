"""Utilities for summarizing multi-run sweeps (YAML top-level lists).

Veeksha CLI supports running multiple `BenchmarkConfig`s in one invocation
via YAML list expansion. This provides a small summarizer to make sweeps
feel "single-run" by writing aggregated artifacts in the parent output directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class SweepRunInfo:
    base_output_dir: str
    run_dir: str
    config_path: str
    metrics_dir: str
    prefill_stats_path: str
    decode_window_metrics_path: str
    summary_stats_path: str
    slo_results_path: str
    throughput_metrics_path: str


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _safe_read_yaml(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _infer_context_length(run: SweepRunInfo) -> Optional[int]:
    # Prefer the prefill_stats grouping key if it is unambiguous.
    prefill = _safe_read_json(run.prefill_stats_path)
    if isinstance(prefill, dict):
        groups = prefill.get("groups")
        if isinstance(groups, dict):
            keys = [k for k in groups.keys() if isinstance(k, str)]
            if len(keys) == 1:
                try:
                    return int(keys[0])
                except Exception:
                    pass

    # Fallback to config.yml for fixed body length generator.
    cfg = _safe_read_yaml(run.config_path)
    if not cfg:
        return None
    try:
        sg = cfg.get("session_generator") or {}
        channels = sg.get("channels") or []
        if not channels or not isinstance(channels, list):
            return None
        text0 = channels[0]
        if not isinstance(text0, dict):
            return None
        blg = text0.get("body_length_generator") or {}
        if not isinstance(blg, dict):
            return None
        if str(blg.get("type", "")).lower() == "fixed" and "value" in blg:
            return int(blg["value"])
    except Exception:
        return None
    return None


def _infer_batch_size(run: SweepRunInfo) -> Optional[int]:
    # Prefer the decode window threshold (what the analysis actually uses).
    dwin = _safe_read_json(run.decode_window_metrics_path)
    if isinstance(dwin, dict):
        cfg = dwin.get("config")
        if isinstance(cfg, dict) and "min_active_requests" in cfg:
            try:
                return int(cfg["min_active_requests"])
            except Exception:
                pass

    # Fallback to traffic scheduler concurrency.
    cfg = _safe_read_yaml(run.config_path)
    if not cfg:
        return None
    try:
        traffic = cfg.get("traffic_scheduler") or {}
        if not isinstance(traffic, dict):
            return None
        if str(traffic.get("type", "")).lower() == "concurrent":
            target = traffic.get("target_concurrent_sessions")
            if target is None:
                return None
            return int(target)
    except Exception:
        return None
    return None


def _infer_output_length(run: SweepRunInfo) -> Optional[int]:
    cfg = _safe_read_yaml(run.config_path)
    if not cfg:
        return None
    try:
        sg = cfg.get("session_generator") or {}
        channels = sg.get("channels") or []
        if not channels or not isinstance(channels, list):
            return None
        text0 = channels[0]
        if not isinstance(text0, dict):
            return None
        olg = text0.get("output_length_generator") or {}
        if not isinstance(olg, dict):
            return None
        if str(olg.get("type", "")).lower() == "fixed" and "value" in olg:
            return int(olg["value"])
    except Exception:
        return None
    return None


def _infer_traffic_knobs(run: SweepRunInfo) -> Dict[str, Any]:
    cfg = _safe_read_yaml(run.config_path) or {}
    traffic = cfg.get("traffic_scheduler") if isinstance(cfg, dict) else None
    if not isinstance(traffic, dict):
        return {}

    traffic_type = str(traffic.get("type", "")).lower()
    out: Dict[str, Any] = {"traffic_type": traffic_type}

    if traffic_type == "concurrent":
        out["target_concurrent_sessions"] = traffic.get("target_concurrent_sessions")
        out["rampup_seconds"] = traffic.get("rampup_seconds")
    elif traffic_type == "rate":
        interval = traffic.get("interval_generator")
        if isinstance(interval, dict):
            out["interval_generator_type"] = str(interval.get("type", "")).lower()
            # common fields
            if "arrival_rate" in interval:
                out["arrival_rate"] = interval.get("arrival_rate")
            if "interval" in interval:
                out["interval_s"] = interval.get("interval")

    return out


def _infer_runtime_knobs(run: SweepRunInfo) -> Dict[str, Any]:
    cfg = _safe_read_yaml(run.config_path) or {}
    runtime = cfg.get("runtime") if isinstance(cfg, dict) else None
    if not isinstance(runtime, dict):
        return {}
    out: Dict[str, Any] = {}
    if "max_sessions" in runtime:
        out["max_sessions"] = runtime.get("max_sessions")
    if "benchmark_timeout" in runtime:
        out["benchmark_timeout_s"] = runtime.get("benchmark_timeout")
    return out


def _extract_prefill_group_stats(
    run: SweepRunInfo, *, prefer_prompt_len: Optional[int]
) -> Dict[str, Any]:
    prefill = _safe_read_json(run.prefill_stats_path)
    if not isinstance(prefill, dict):
        return {}
    groups = prefill.get("groups")
    if not isinstance(groups, dict) or not groups:
        return {}

    chosen_key: Optional[str] = None
    if prefer_prompt_len is not None and str(prefer_prompt_len) in groups:
        chosen_key = str(prefer_prompt_len)
    else:
        keys = [k for k in groups.keys() if isinstance(k, str)]
        if len(keys) == 1:
            chosen_key = keys[0]

    if chosen_key is None:
        return {}
    entry = groups.get(chosen_key)
    if not isinstance(entry, dict):
        return {}
    out: Dict[str, Any] = {"prompt_len": int(chosen_key)}
    for k in ("count", "mean", "median", "p90", "p99", "min", "max", "std"):
        if k in entry:
            out[k] = entry.get(k)
    out["group_by"] = prefill.get("group_by")
    return out


def _extract_decode_window_stats(run: SweepRunInfo) -> Dict[str, Any]:
    dwin = _safe_read_json(run.decode_window_metrics_path)
    if not isinstance(dwin, dict):
        return {}
    stats = dwin.get("tbc_in_window_stats")
    if not isinstance(stats, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in ("count", "mean", "median", "p90", "p99", "min", "max", "std"):
        if k in stats:
            out[k] = stats.get(k)
    # Support both new 'windows' structure and legacy 'window' structure
    windows = dwin.get("windows")
    if isinstance(windows, dict):
        out["window_duration_s"] = windows.get("total_duration_s")
        out["num_windows"] = windows.get("num_selected_segments")
    else:
        # Legacy fallback
        window = dwin.get("window")
        if isinstance(window, dict):
            out["window_duration_s"] = window.get("duration_s")
    cfg = dwin.get("config")
    if isinstance(cfg, dict) and "min_active_requests" in cfg:
        out["min_active_requests"] = cfg.get("min_active_requests")
    return out


def _extract_slo_summary(run: SweepRunInfo) -> Dict[str, Any]:
    slo = _safe_read_json(run.slo_results_path)
    if not isinstance(slo, dict):
        return {}
    out: Dict[str, Any] = {}
    if "all_slos_met" in slo:
        out["all_slos_met"] = bool(slo.get("all_slos_met"))
    # Keep the full result available in JSON summary, but don't try to flatten it
    # into the CSV table.
    out["slo_results"] = slo
    return out


def _extract_throughput_metrics(run: SweepRunInfo) -> Dict[str, Any]:
    tm = _safe_read_json(run.throughput_metrics_path)
    if not isinstance(tm, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in ("tpot_based_throughput", "tbc_based_throughput"):
        if k in tm:
            out[k] = tm.get(k)
    return out


def _extract_summary_stats(run: SweepRunInfo) -> Dict[str, Any]:
    ss = _safe_read_json(run.summary_stats_path)
    return ss if isinstance(ss, dict) else {}


def _write_csv(path: str, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers))
        f.write("\n")
        for row in rows:
            f.write(
                ",".join("" if row.get(h) is None else str(row.get(h)) for h in headers)
            )
            f.write("\n")


def write_sweep_summary(base_output_dir: str, run_dirs: list[str]) -> Dict[str, str]:
    """Write sweep-level summary artifacts under the parent output directory.

    Writes:
    - `sweep_manifest.json`: list of run directories included in the sweep.
    - `sweep_summary.json`: per-run summary including common performance artifacts.
    - `sweep_summary.csv`: table view of common per-run metrics for spreadsheet/W&B.
    - `decode_stats.json`: aggregated decode-window stats if present.
    - `decode_stats.csv`: a simple CSV view of the decode-window stats if present.

    Args:
        base_output_dir: The parent output directory shared by all runs.
        run_dirs: Resolved output directories for each completed run.

    Returns:
        Dict with keys for written artifact paths.
    """
    os.makedirs(base_output_dir, exist_ok=True)

    runs: list[SweepRunInfo] = []
    for rd in run_dirs:
        runs.append(
            SweepRunInfo(
                base_output_dir=base_output_dir,
                run_dir=rd,
                config_path=os.path.join(rd, "config.yml"),
                metrics_dir=os.path.join(rd, "metrics"),
                prefill_stats_path=os.path.join(rd, "metrics", "prefill_stats.json"),
                decode_window_metrics_path=os.path.join(
                    rd, "metrics", "decode_window_metrics.json"
                ),
                summary_stats_path=os.path.join(rd, "metrics", "summary_stats.json"),
                slo_results_path=os.path.join(rd, "metrics", "slo_results.json"),
                throughput_metrics_path=os.path.join(
                    rd, "metrics", "throughput_metrics.json"
                ),
            )
        )

    # Write sweep files directly to base_output_dir (which is already the sweep directory)
    manifest = {
        "sweep_dir": base_output_dir,
        "num_runs": len(runs),
        "runs": [
            {
                "run_dir": r.run_dir,
                "config_path": r.config_path,
                "metrics_dir": r.metrics_dir,
                "prefill_stats_path": r.prefill_stats_path,
                "decode_window_metrics_path": r.decode_window_metrics_path,
            }
            for r in runs
        ],
    }
    manifest_path = os.path.join(base_output_dir, "sweep_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # General per-run summary table (works for decode sweeps and other perf sweeps).
    per_run_rows: list[dict[str, Any]] = []
    per_run_json: list[dict[str, Any]] = []

    for idx, run in enumerate(runs):
        cl = _infer_context_length(run)
        bs = _infer_batch_size(run)
        ol = _infer_output_length(run)
        traffic = _infer_traffic_knobs(run)
        runtime = _infer_runtime_knobs(run)

        summary_stats = _extract_summary_stats(run)
        slo_summary = _extract_slo_summary(run)
        throughput = _extract_throughput_metrics(run)
        prefill_group = _extract_prefill_group_stats(run, prefer_prompt_len=cl)
        decode_window = _extract_decode_window_stats(run)

        record = {
            "run_index": idx,
            "run_dir": run.run_dir,
            "context_length": cl,
            "batch_size": bs,
            "output_length": ol,
            "traffic": traffic,
            "runtime": runtime,
            "summary_stats": summary_stats,
            "throughput_metrics": throughput,
            "prefill_group_stats": prefill_group,
            "decode_window_stats": decode_window,
            **slo_summary,
        }
        per_run_json.append(record)

        per_run_rows.append(
            {
                "run_index": idx,
                "run_dir": run.run_dir,
                "traffic_type": traffic.get("traffic_type"),
                "target_concurrent_sessions": traffic.get("target_concurrent_sessions"),
                "arrival_rate": traffic.get("arrival_rate"),
                "context_length": cl,
                "output_length": ol,
                "max_sessions": runtime.get("max_sessions"),
                "benchmark_timeout_s": runtime.get("benchmark_timeout_s"),
                "num_requests": summary_stats.get("Number of Requests"),
                "num_completed_requests": summary_stats.get(
                    "Number of Completed Requests"
                ),
                "error_rate": summary_stats.get("Error Rate"),
                "all_slos_met": slo_summary.get("all_slos_met"),
                "tpot_based_throughput": throughput.get("tpot_based_throughput"),
                "tbc_based_throughput": throughput.get("tbc_based_throughput"),
            }
        )

    sweep_summary_path = os.path.join(base_output_dir, "sweep_summary.json")
    with open(sweep_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_output_dir": base_output_dir,
                "num_runs": len(runs),
                "runs": per_run_json,
            },
            f,
            indent=2,
        )

    sweep_csv_path = os.path.join(base_output_dir, "sweep_summary.csv")
    sweep_headers = [
        "run_index",
        "run_dir",
        "traffic_type",
        "target_concurrent_sessions",
        "arrival_rate",
        "context_length",
        "output_length",
        "max_sessions",
        "benchmark_timeout_s",
        "num_requests",
        "num_completed_requests",
        "error_rate",
        "all_slos_met",
        "tpot_based_throughput",
        "tbc_based_throughput",
    ]
    _write_csv(sweep_csv_path, sweep_headers, per_run_rows)

    written: Dict[str, str] = {"sweep_manifest": manifest_path}
    written["sweep_summary"] = sweep_summary_path
    written["sweep_summary_csv"] = sweep_csv_path

    # Decode-window aggregation (optional): key by <context_length>_<batch_size>.
    decode_stats: dict[str, Any] = {}
    decode_rows: list[dict[str, Any]] = []

    for run in runs:
        cl = _infer_context_length(run)
        bs = _infer_batch_size(run)
        if cl is None or bs is None:
            continue
        dwin = _safe_read_json(run.decode_window_metrics_path)
        if not isinstance(dwin, dict) or not dwin:
            continue

        key = f"{cl}_{bs}"
        decode_stats[key] = dwin

        tbc_stats = dwin.get("tbc_in_window_stats")
        tbc_count = None
        tbc_mean = None
        tbc_p99 = None
        if isinstance(tbc_stats, dict):
            tbc_count = tbc_stats.get("count")
            tbc_mean = tbc_stats.get("mean")
            tbc_p99 = tbc_stats.get("p99")

        window_start = None
        window_end = None
        window_duration_s = None
        if isinstance(dwin.get("window"), dict):
            w = dwin["window"]
            window_start = w.get("start")
            window_end = w.get("end")
            window_duration_s = w.get("duration_s")
        elif isinstance(dwin.get("windows"), dict):
            ww = dwin["windows"]
            window_duration_s = ww.get("total_duration_s")
            per = ww.get("per_window")
            if isinstance(per, list) and per and isinstance(per[0], dict):
                window_start = per[0].get("start")
                window_end = per[0].get("end")

        decode_rows.append(
            {
                "key": key,
                "run_dir": run.run_dir,
                "context_length": cl,
                "batch_size": bs,
                "window_start": window_start,
                "window_end": window_end,
                "window_duration_s": window_duration_s,
                "tbc_count": tbc_count,
                "tbc_mean": tbc_mean,
                "tbc_p99": tbc_p99,
            }
        )

    if decode_stats:
        decode_stats_path = os.path.join(base_output_dir, "decode_stats.json")
        with open(decode_stats_path, "w", encoding="utf-8") as f:
            json.dump(decode_stats, f, indent=2)
        written["decode_stats"] = decode_stats_path

        decode_csv_path = os.path.join(base_output_dir, "decode_stats.csv")
        decode_headers = [
            "key",
            "run_dir",
            "context_length",
            "batch_size",
            "window_start",
            "window_end",
            "window_duration_s",
            "tbc_count",
            "tbc_mean",
            "tbc_p99",
        ]
        _write_csv(decode_csv_path, decode_headers, decode_rows)
        written["decode_stats_csv"] = decode_csv_path

    return written
