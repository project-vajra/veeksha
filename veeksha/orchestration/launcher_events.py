"""Launcher event payloads and console messages for orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from veeksha.orchestration.launcher_progress import (
    positive_int_or_none,
    progress_percentage,
)


def sweep_plan_payload(plan: Any, generated_config_dir: Path) -> dict:
    payload = {
        "runs": len(plan.runs),
        "generated_config_dir": str(generated_config_dir),
        "sweep_type": plan.spec.sweep_type,
        "engine": plan.spec.engine,
        "model": plan.spec.model,
        "trace": plan.runs[0].trace_source if plan.runs else "unknown",
        "base_config": str(plan.base_config_path),
        "concurrencies": sorted({run.concurrency for run in plan.runs}),
        "input_sizes": sorted(
            {run.input_size for run in plan.runs if run.input_size is not None}
        ),
    }
    if getattr(plan, "endpoint", None) is not None:
        payload["endpoint"] = plan.endpoint.to_dict()
    return payload


def run_event_payload(spec: Any, run: Any, *, completed_runs: int) -> dict:
    completed_runs = min(max(completed_runs, 0), run.run_count)
    return {
        "sweep_type": spec.sweep_type,
        "engine": spec.engine,
        "model": spec.model,
        "trace": run.trace_source,
        "run_index": run.run_index,
        "run_count": run.run_count,
        "run_name": run.run_name,
        "concurrency": run.concurrency,
        "input_size": run.input_size,
        "completed_runs": completed_runs,
        "remaining_runs": max(run.run_count - completed_runs, 0),
        "sweep_progress_pct": progress_percentage(completed_runs, run.run_count),
    }


def console_message(event: str, payload: dict) -> Optional[str]:
    if event == "launcher_start":
        return (
            f"output={payload['output_dir']} events={payload['events_file']} "
            f"benchmark_logs={payload['benchmark_log_dir']}"
        )
    if event == "sweep_plan_ready":
        dimensions = _plan_dimensions(payload)
        if dimensions:
            dimensions = f" {dimensions}"
        return (
            f"sweep plan ready: {_sweep_details(payload)} runs={payload['runs']}"
            f"{dimensions} generated_configs={payload['generated_config_dir']} "
            f"base_config={payload['base_config']}"
        )
    if event == "engine_unmanaged":
        return "no server or endpoint configured; running sweep against config api_base"
    if event == "endpoint_external":
        return f"using external endpoint: {_endpoint_details(payload)}"
    if event == "engine_start":
        return f"starting engine: {_engine_details(payload)}"
    if event == "engine_ready":
        return f"engine ready: api_base={payload['api_base']}"
    if event == "engine_restart":
        return (
            f"restarting engine: reason={payload['reason']} {_engine_details(payload)}"
        )
    if event == "engine_restart_exhausted":
        return (
            f"engine restart budget exhausted: reason={payload['reason']} "
            f"{_engine_details(payload)}"
        )
    if event == "engine_stop":
        return f"stopping engine: {_engine_details(payload)}"
    if event == "engine_stopped":
        return "engine stopped"
    if event == "benchmark_attempt_start":
        return (
            f"run {payload['run_index']}/{payload['run_count']} attempt "
            f"{payload['attempt']} starting: {_run_details(payload)} "
            f"sweep_progress={_sweep_progress(payload)} config={payload['config']} "
            f"stdout={payload['stdout_log']} stderr={payload['stderr_log']}"
        )
    if event == "benchmark_attempt_progress":
        return (
            f"run {payload['run_index']}/{payload['run_count']} attempt "
            f"{payload['attempt']} running: "
            f"requests_processed={_request_progress(payload)} "
            f"sweep_progress={_sweep_progress(payload)}"
        )
    if event == "benchmark_attempt_success":
        return (
            f"run {payload['run_index']}/{payload['run_count']} attempt "
            f"{payload['attempt']} succeeded rc={payload['returncode']}; "
            f"requests_processed={_request_progress(payload)} "
            f"sweep_progress={_sweep_progress(payload)}"
        )
    if event == "benchmark_attempt_failed":
        return (
            f"run {payload['run_index']}/{payload['run_count']} attempt "
            f"{payload['attempt']} failed reason={payload['reason']} "
            f"rc={payload.get('returncode')}; "
            f"requests_processed={_request_progress(payload)} "
            f"sweep_progress={_sweep_progress(payload)}"
        )
    if event == "benchmark_attempts_exhausted":
        return f"{payload['message']}; sweep_progress={_sweep_progress(payload)}"
    if event == "cooldown_start":
        return (
            f"cooldown before next run: {payload['seconds']}s after "
            f"run {payload['run_index']}/{payload['run_count']}; "
            f"sweep_progress={_sweep_progress(payload)} "
            f"next_run={payload['next_run_index']}/{payload['run_count']}"
        )
    if event == "sweep_complete":
        return f"sweep complete: {payload['runs']} run(s)"
    return None


def attempt_log_name(run_index: int, attempt: int, suffix: str) -> str:
    return f"run_{run_index:03d}_attempt_{attempt:02d}_{suffix}"


def _sweep_details(payload: dict) -> str:
    return (
        f"sweep={payload['sweep_type']} engine={payload['engine']} "
        f"model={payload['model']} trace={payload['trace']}"
    )


def _plan_dimensions(payload: dict) -> str:
    parts = []
    if payload.get("concurrencies"):
        parts.append(f"concurrencies={_format_values(payload['concurrencies'])}")
    if payload.get("input_sizes"):
        parts.append(f"input_sizes={_format_values(payload['input_sizes'])}")
    return " ".join(parts)


def _run_details(payload: dict) -> str:
    parts = [
        _sweep_details(payload),
        f"run={payload['run_name']}",
        f"concurrency={payload['concurrency']}",
    ]
    if payload.get("input_size") is not None:
        parts.append(f"input_size={payload['input_size']}")
    return " ".join(parts)


def _sweep_progress(payload: dict) -> str:
    return (
        f"{payload['completed_runs']}/{payload['run_count']} complete "
        f"({float(payload['sweep_progress_pct']):.1f}%)"
    )


def _request_progress(payload: dict) -> str:
    completed_requests = int(payload["requests_completed"])
    request_total = positive_int_or_none(payload.get("request_total"))
    if request_total is None:
        return str(completed_requests)
    request_pct = payload.get("request_progress_pct")
    if request_pct is None:
        request_pct = progress_percentage(completed_requests, request_total)
    return f"{completed_requests}/{request_total} ({float(request_pct):.1f}%)"


def _format_values(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _engine_details(payload: dict) -> str:
    parts = [
        f"runner={payload['runner']}",
        f"engine_type={payload.get('engine_type', 'unknown')}",
        f"model={payload.get('model', 'unknown')}",
        f"api_base={payload['api_base']}",
        f"health={payload['health_url']}",
        f"logs={payload['engine_log_dir']}",
    ]
    if payload.get("container"):
        parts.append(f"container={payload['container']}")
    return " ".join(parts)


def _endpoint_details(payload: dict) -> str:
    parts = [
        f"engine_type={payload.get('engine_type', 'unknown')}",
        f"model={payload.get('model', 'unknown')}",
        f"api_base={payload['api_base']}",
    ]
    if payload.get("health_url") is not None:
        parts.append(f"health={payload['health_url']}")
    if payload.get("port"):
        parts.append(f"host={payload.get('host', 'unknown')}")
        parts.append(f"port={payload['port']}")
    return " ".join(parts)
