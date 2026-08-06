"""Compile and run provider-independent named benchmarks with Veeksha."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from vidhi import create_class_from_dict

from veeksha.benchmark import manage_benchmark_run, run_benchmark_with_endpoint
from veeksha.cli.parsing import parse_cli_sweep
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.named_benchmark import NamedBenchmarkConfig
from veeksha.config.runtime import RuntimeConfig
from veeksha.config.utils import redact_config_secrets, to_serializable_config_dict
from veeksha.logger import init_logger
from veeksha.named_benchmarks import (
    Benchmark,
    ConcurrencyLoadPoint,
    DatasetCase,
    Modality,
    load_benchmark,
)
from veeksha.named_benchmarks.aggregation import (
    CompletedDatasetRun,
    build_named_benchmark_results,
    build_named_benchmark_sweep_results,
)
from veeksha.orchestration.benchmark_orchestrator import managed_server

logger = init_logger(__name__)

_DATASET_ROOT_PLACEHOLDER = "${DATASET_ROOT}"
_PREPARATION_METADATA_NAME = "preparation.json"
_SHA256_RE = re.compile(r"(?:sha256:)?([0-9a-fA-F]{64})")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class CompiledDatasetRun:
    """One ordinary Veeksha run materialized from a dataset and target."""

    target_id: str
    dataset_id: str
    config: BenchmarkConfig
    config_path: Path
    load_point: ConcurrencyLoadPoint | None = None
    expected_session_count: int | None = None


@dataclass(frozen=True, slots=True)
class NamedBenchmarkPlan:
    benchmark: Benchmark
    parent_dir: Path
    targets: tuple[dict[str, Any], ...]
    children: tuple[CompiledDatasetRun, ...]


def _benchmark_load_points(
    benchmark: Benchmark,
) -> tuple[ConcurrencyLoadPoint | None, ...]:
    load = benchmark.execution.load
    return (None,) if load is None else load.points


def _compiled_config_path(
    *,
    compiled_dir: Path,
    target_id: str,
    dataset_id: str,
    load_point: ConcurrencyLoadPoint | None,
) -> Path:
    if load_point is None:
        return compiled_dir / target_id / f"{dataset_id}.yml"
    return compiled_dir / target_id / "loads" / load_point.id / f"{dataset_id}.yml"


def _child_output_dir(
    *,
    parent_dir: Path,
    target_id: str,
    dataset_id: str,
    load_point: ConcurrencyLoadPoint | None,
) -> Path:
    target_dir = parent_dir / "targets" / target_id
    if load_point is None:
        return target_dir / "datasets" / dataset_id
    return target_dir / "loads" / load_point.id / "datasets" / dataset_id


def _validate_load_capacity(
    benchmark: Benchmark,
) -> None:
    load = benchmark.execution.load
    if load is None:
        return
    required_sessions = max(load.values)
    runtime = benchmark.execution.config.get("runtime", {})
    runtime_limit = (
        runtime.get("max_sessions", RuntimeConfig().max_sessions)
        if isinstance(runtime, Mapping)
        else RuntimeConfig().max_sessions
    )

    for dataset in benchmark.datasets:
        capacity_limits: list[tuple[str, int]] = []
        if type(runtime_limit) is int and runtime_limit >= 0:
            capacity_limits.append(("execution.runtime.max_sessions", runtime_limit))
        if (
            dataset.session_generator.get("wrap_mode", False) is not True
            and dataset.source.expected_rows is not None
        ):
            capacity_limits.append(
                ("dataset.source.expected_rows", dataset.source.expected_rows)
            )

        for source, available_sessions in capacity_limits:
            if available_sessions < required_sessions:
                raise ValueError(
                    f"benchmark={benchmark.id}, dataset={dataset.id}: maximum "
                    f"concurrency {required_sessions} cannot be reached because "
                    f"{source} is {available_sessions}"
                )


def compile_named_benchmark(config: NamedBenchmarkConfig) -> NamedBenchmarkPlan:
    """Resolve a catalog entry and target sweep into dataset-level runs."""

    runner_state = _runner_state(Path(config.target_config))
    benchmark = load_benchmark(config.benchmark)
    _validate_load_capacity(benchmark)
    _validate_materialized_dataset_provenance(
        benchmark,
        dataset_root=config.dataset_root,
    )
    target_configs = parse_cli_sweep(
        BenchmarkConfig,
        args=["--config", config.target_config],
    )
    if not target_configs:
        raise ValueError(f"No targets resolved from {config.target_config!r}")

    target_configs, target_bindings = _deduplicate_target_configs(target_configs)
    target_ids = _unique_target_ids(target_bindings)
    parent_dir = _create_parent_dir(config, benchmark, target_bindings)
    compiled_dir = parent_dir / "compiled"

    load_points = _benchmark_load_points(benchmark)
    children: list[CompiledDatasetRun] = []
    target_records: list[dict[str, Any]] = []
    for target_id, target, binding in zip(
        target_ids, target_configs, target_bindings, strict=True
    ):
        target_records.append(
            {
                "target_id": target_id,
                "binding": binding,
            }
        )
        for load_point in load_points:
            for dataset in benchmark.datasets:
                child_config = _compile_child_config(
                    benchmark=benchmark,
                    dataset=dataset,
                    target=target,
                    parent_dir=parent_dir,
                    target_id=target_id,
                    dataset_root=config.dataset_root,
                    load_point=load_point,
                )
                config_path = _compiled_config_path(
                    compiled_dir=compiled_dir,
                    target_id=target_id,
                    dataset_id=dataset.id,
                    load_point=load_point,
                )
                _write_yaml(
                    config_path,
                    redact_config_secrets(to_serializable_config_dict(child_config)),
                )
                children.append(
                    CompiledDatasetRun(
                        target_id=target_id,
                        dataset_id=dataset.id,
                        config=child_config,
                        config_path=config_path,
                        load_point=load_point,
                        expected_session_count=_expected_session_count(
                            dataset,
                            child_config,
                        ),
                    )
                )

    plan = NamedBenchmarkPlan(
        benchmark=benchmark,
        parent_dir=parent_dir,
        targets=tuple(target_records),
        children=tuple(children),
    )
    _write_plan_manifest(plan, config, runner_state=runner_state)
    return plan


def run_named_benchmark(config: NamedBenchmarkConfig) -> Path:
    """Compile and, unless requested otherwise, execute one named benchmark."""

    plan = compile_named_benchmark(config)
    logger.info(
        "Compiled %s into %d dataset/target run(s) at %s",
        plan.benchmark.id,
        len(plan.children),
        plan.parent_dir,
    )
    if config.dry_run:
        _write_json(
            plan.parent_dir / "run_status.json",
            {
                "status": "dry_run",
                "compiled_runs": len(plan.children),
                "completed_runs": 0,
                "failed_runs": 0,
            },
        )
        return plan.parent_dir

    completed: list[CompletedDatasetRun] = []
    failures: list[dict[str, Any]] = []
    children_by_target: dict[str, list[CompiledDatasetRun]] = {}
    for child in plan.children:
        children_by_target.setdefault(child.target_id, []).append(child)

    for target_id in (record["target_id"] for record in plan.targets):
        target_children = children_by_target[target_id]
        server = target_children[0].config.server
        if server is None:
            for child in target_children:
                _run_child(child, completed, failures)
            continue

        server_dir = plan.parent_dir / "targets" / target_id / "managed_server"
        try:
            with managed_server(server, output_dir=str(server_dir)) as server_info:
                endpoint = server_info["endpoint"]
                for child in target_children:
                    _run_child(child, completed, failures, endpoint=endpoint)
        except Exception as exc:
            logger.exception("Managed target %s failed", target_id)
            already_recorded = {
                (
                    failure["target_id"],
                    failure.get("load_point_id"),
                    failure["dataset_id"],
                )
                for failure in failures
            }
            already_completed = {
                (child.target_id, child.load_point_id, child.dataset_id)
                for child in completed
            }
            for child in target_children:
                key = _compiled_child_key(child)
                if key not in already_recorded and key not in already_completed:
                    failures.append(_child_failure(child, exc))

    _write_results(plan, completed, failures)
    if failures:
        failed_pairs = ", ".join(_failure_label(item) for item in failures)
        raise RuntimeError(
            f"Named benchmark {plan.benchmark.id} completed with "
            f"{len(failures)} failed child run(s): {failed_pairs}. "
            f"Partial results are preserved at {plan.parent_dir}."
        )
    return plan.parent_dir


def run_cli(configs: Sequence[NamedBenchmarkConfig]) -> list[Path]:
    """CLI adapter for one or more named benchmark configurations."""

    return [run_named_benchmark(config) for config in configs]


def _compile_child_config(
    *,
    benchmark: Benchmark,
    dataset: DatasetCase,
    target: BenchmarkConfig,
    parent_dir: Path,
    target_id: str,
    dataset_root: str,
    load_point: ConcurrencyLoadPoint | None,
) -> BenchmarkConfig:
    target_dict = to_serializable_config_dict(target)
    client = copy.deepcopy(target_dict["client"])
    client = _deep_merge(client, benchmark.client_overrides)
    client = _deep_merge(client, dataset.client_overrides)

    child_data = copy.deepcopy(benchmark.execution.config)
    child_data["client"] = client
    child_data["session_generator"] = copy.deepcopy(dataset.session_generator)
    child_data["output_dir"] = str(
        _child_output_dir(
            parent_dir=parent_dir,
            target_id=target_id,
            dataset_id=dataset.id,
            load_point=load_point,
        )
    )
    if target_dict.get("server") is not None:
        child_data["server"] = copy.deepcopy(target_dict["server"])
    if target_dict.get("endpoint") is not None:
        child_data["endpoint"] = copy.deepcopy(target_dict["endpoint"])

    if load_point is not None:
        traffic = child_data["traffic_scheduler"]
        traffic["target_concurrent_sessions"] = load_point.target_concurrent_sessions

    child_data = _expand_dataset_root(child_data, dataset_root)
    child = create_class_from_dict(BenchmarkConfig, child_data)
    _validate_interaction_contract(benchmark, child, dataset.id)
    return child


def _expected_session_count(
    dataset: DatasetCase,
    child_config: BenchmarkConfig,
) -> int | None:
    """Return the exact number of sessions a complete child should evaluate."""

    runtime_limit = child_config.runtime.max_sessions
    source_count = dataset.source.expected_rows
    wrap_mode = dataset.session_generator.get("wrap_mode", False) is True

    if runtime_limit >= 0:
        if source_count is not None and not wrap_mode:
            return min(runtime_limit, source_count)
        return runtime_limit
    if not wrap_mode:
        return source_count
    return None


def _validate_interaction_contract(
    benchmark: Benchmark, child: BenchmarkConfig, dataset_id: str
) -> None:
    child_data = to_serializable_config_dict(child)
    client = child_data["client"]
    client_type = str(client.get("type", ""))
    context = f"benchmark={benchmark.id}, dataset={dataset_id}"

    if benchmark.modality is Modality.ASR:
        if client_type != "stt":
            raise ValueError(
                f"{context}: ASR requires client.type=stt, got {client_type}"
            )
        realtime = bool(client.get("ws_realtime_pacing", False))
        expected_realtime = benchmark.input_mode.value == "streaming"
        if realtime != expected_realtime:
            raise ValueError(
                f"{context}: input_mode={benchmark.input_mode.value} requires "
                f"client.ws_realtime_pacing={expected_realtime}"
            )
        _validate_language_contract(
            benchmark,
            client=client,
            context=context,
        )
        _validate_asr_audio_contract(benchmark, client=client, context=context)
        return

    if client_type not in {"tts", "streaming_tts"}:
        raise ValueError(
            f"{context}: TTS requires client.type=tts or streaming_tts, got "
            f"{client_type}"
        )
    transport = benchmark.interaction.get("transport")
    if transport == "websocket" and client_type != "streaming_tts":
        raise ValueError(
            f"{context}: interaction.transport=websocket requires "
            "client.type=streaming_tts"
        )
    if client_type == "tts":
        actual_input_mode = "static"
    else:
        actual_input_mode = str(client.get("input_output_mode", "complete_text"))
        actual_input_mode = "streaming" if actual_input_mode == "duplex" else "static"
    if benchmark.input_mode.value != actual_input_mode:
        raise ValueError(
            f"{context}: input_mode={benchmark.input_mode.value} requires "
            "client.type=tts or streaming_tts with the matching "
            "input_output_mode; got "
            f"client.type={client_type}, input_output_mode="
            f"{client.get('input_output_mode')!r}"
        )
    _validate_language_contract(benchmark, client=client, context=context)
    _validate_tts_audio_contract(
        benchmark,
        child_data=child_data,
        client=client,
        context=context,
    )


def _validate_language_contract(
    benchmark: Benchmark,
    *,
    client: Mapping[str, Any],
    context: str,
) -> None:
    required_value = benchmark.interaction.get("required_languages")
    if required_value is None:
        return
    if (
        not isinstance(required_value, Sequence)
        or isinstance(required_value, (str, bytes))
        or not required_value
        or any(
            not isinstance(value, str) or not value.strip() for value in required_value
        )
    ):
        raise ValueError(
            f"{context}: interaction.required_languages must be a non-empty "
            "list of language codes"
        )

    required_by_code = {
        value.strip().casefold(): value.strip() for value in required_value
    }
    if len(required_by_code) != len(required_value):
        raise ValueError(
            f"{context}: interaction.required_languages contains duplicate codes"
        )
    language_mode = client.get("language_mode")
    if language_mode not in {"request_metadata", "auto"}:
        raise ValueError(
            f"{context}: multilingual benchmark requires "
            "client.language_mode=request_metadata or auto"
        )
    if language_mode == "request_metadata":
        metadata_key = benchmark.interaction.get("language_metadata_key", "language")
        if client.get("language_metadata_key") != metadata_key:
            raise ValueError(
                f"{context}: client.language_metadata_key must be " f"{metadata_key!r}"
            )

    supported_value = client.get("supported_languages")
    if not isinstance(supported_value, Sequence) or isinstance(
        supported_value, (str, bytes)
    ):
        supported_value = ()
    supported = {
        value.strip().casefold()
        for value in supported_value
        if isinstance(value, str) and value.strip()
    }
    missing = sorted(
        required_by_code[code] for code in required_by_code.keys() - supported
    )
    if missing:
        raise ValueError(
            f"{context}: client.supported_languages does not cover required "
            f"languages: {', '.join(missing)}"
        )


def _validate_asr_audio_contract(
    benchmark: Benchmark,
    *,
    client: Mapping[str, Any],
    context: str,
) -> None:
    interaction = benchmark.interaction
    expected_sample_rate = interaction.get("sample_rate_hz")
    if (
        expected_sample_rate is not None
        and client.get("sample_rate") != expected_sample_rate
    ):
        raise ValueError(
            f"{context}: client.sample_rate={client.get('sample_rate')!r}; "
            f"benchmark requires {expected_sample_rate} Hz"
        )
    frame_ms = interaction.get("input_frame_ms")
    if frame_ms is not None and expected_sample_rate is not None:
        expected_chunk_size = expected_sample_rate * frame_ms * 2 / 1000
        if not expected_chunk_size.is_integer():
            raise ValueError(
                f"{context}: input_frame_ms does not resolve to whole PCM16 bytes"
            )
        if client.get("ws_chunk_size") != int(expected_chunk_size):
            raise ValueError(
                f"{context}: client.ws_chunk_size={client.get('ws_chunk_size')!r}; "
                f"benchmark requires {int(expected_chunk_size)} bytes"
            )


def _validate_tts_audio_contract(
    benchmark: Benchmark,
    *,
    child_data: Mapping[str, Any],
    client: Mapping[str, Any],
    context: str,
) -> None:
    interaction = benchmark.interaction
    encoding = interaction.get("output_audio_encoding")
    if encoding is not None and encoding != "pcm_s16le":
        raise ValueError(
            f"{context}: named streaming TTS currently supports only pcm_s16le, "
            f"got {encoding!r}"
        )
    expected_channels = interaction.get("output_channels")
    if expected_channels is not None and expected_channels != 1:
        raise ValueError(
            f"{context}: named streaming TTS currently supports mono output only"
        )
    expected_sample_rate = interaction.get("output_sample_rate_hz")
    if (
        expected_sample_rate is not None
        and client.get("sample_rate") != expected_sample_rate
    ):
        raise ValueError(
            f"{context}: client.sample_rate={client.get('sample_rate')!r}; "
            f"benchmark requires {expected_sample_rate} Hz"
        )
    if encoding == "pcm_s16le" and not client.get("strict_audio_contract", False):
        raise ValueError(
            f"{context}: raw PCM benchmark requires client.strict_audio_contract=true"
        )

    frame_ms = interaction.get("playable_frame_ms")
    if frame_ms is None:
        return
    audio_channel = _performance_audio_channel(child_data)
    if not audio_channel.get("interactivity_enabled", False):
        raise ValueError(
            f"{context}: playable-frame metrics require audio interactivity evaluation"
        )
    if audio_channel.get("fluidity_frame_ms") != frame_ms:
        raise ValueError(
            f"{context}: evaluator fluidity_frame_ms="
            f"{audio_channel.get('fluidity_frame_ms')!r}; benchmark requires "
            f"{frame_ms}"
        )


def _performance_audio_channel(child_data: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluators = child_data.get("evaluators")
    if not isinstance(evaluators, Sequence) or isinstance(evaluators, (str, bytes)):
        return {}
    for evaluator in evaluators:
        if not isinstance(evaluator, Mapping) or evaluator.get("type") != "performance":
            continue
        audio_channel = evaluator.get("audio_channel")
        if isinstance(audio_channel, Mapping):
            return audio_channel
    return {}


def _load_point_payload(
    load_point: ConcurrencyLoadPoint | None,
) -> dict[str, Any] | None:
    return None if load_point is None else load_point.to_mapping()


def _compiled_child_key(
    child: CompiledDatasetRun,
) -> tuple[str, str | None, str]:
    return (
        child.target_id,
        None if child.load_point is None else child.load_point.id,
        child.dataset_id,
    )


def _child_failure(
    child: CompiledDatasetRun,
    exc: Exception,
) -> dict[str, Any]:
    failure: dict[str, Any] = {
        "target_id": child.target_id,
        "dataset_id": child.dataset_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if child.load_point is not None:
        failure["load_point_id"] = child.load_point.id
        failure["load"] = child.load_point.to_mapping()
    return failure


def _failure_label(failure: Mapping[str, Any]) -> str:
    load_point_id = failure.get("load_point_id")
    parts = [str(failure["target_id"])]
    if load_point_id is not None:
        parts.append(str(load_point_id))
    parts.append(str(failure["dataset_id"]))
    return "/".join(parts)


def _read_request_metric_rows(run_dir: str | Path) -> list[Mapping[str, Any]]:
    path = Path(run_dir) / "metrics" / "request_level_metrics.jsonl"
    if not path.is_file():
        return []
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"Expected a JSON object in {path} at line {line_number}."
                )
            rows.append(row)
    return rows


def _finite_timestamp(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return timestamp if math.isfinite(timestamp) else None


def _session_intervals(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int | None, list[tuple[float, float]]]:
    session_ids: set[str] = set()
    starts_by_session: dict[str, list[float]] = {}
    ends_by_session: dict[str, list[float]] = {}
    missing_session_id = False

    for row in rows:
        raw_session_id = row.get("session_id")
        if raw_session_id is None:
            missing_session_id = True
            continue
        session_id = str(raw_session_id)
        session_ids.add(session_id)
        start = _finite_timestamp(row.get("scheduler_dispatched_at"))
        end = _finite_timestamp(row.get("client_completed_at"))
        if start is not None:
            starts_by_session.setdefault(session_id, []).append(start)
        if end is not None:
            ends_by_session.setdefault(session_id, []).append(end)

    intervals: list[tuple[float, float]] = []
    for session_id in sorted(session_ids):
        starts = starts_by_session.get(session_id, [])
        ends = ends_by_session.get(session_id, [])
        if not starts or not ends:
            continue
        start = min(starts)
        end = max(ends)
        if end > start:
            intervals.append((start, end))

    observed_count = None if missing_session_id else len(session_ids)
    return observed_count, intervals


def _measure_achieved_concurrency(
    intervals: Sequence[tuple[float, float]],
    *,
    target: int,
    rampup_seconds: float,
) -> dict[str, Any]:
    if not intervals:
        return {
            "target_concurrent_sessions": target,
            "rampup_seconds": rampup_seconds,
            "max_observed_concurrency": 0,
            "max_observed_steady_state_concurrency": 0,
            "steady_state_duration_seconds": 0.0,
            "seconds_at_or_above_target": 0.0,
            "steady_state_target_coverage": 0.0,
            "target_achieved": False,
        }

    events: dict[float, int] = {}
    for start, end in intervals:
        events[start] = events.get(start, 0) + 1
        events[end] = events.get(end, 0) - 1

    times = sorted(events)
    benchmark_start = min(start for start, _ in intervals)
    benchmark_end = max(end for _, end in intervals)
    steady_start = benchmark_start + rampup_seconds
    current = 0
    max_observed = 0
    max_steady = 0
    seconds_at_target = 0.0

    for index, timestamp in enumerate(times):
        current += events[timestamp]
        max_observed = max(max_observed, current)
        if index + 1 == len(times):
            continue
        interval_end = times[index + 1]
        if interval_end <= timestamp:
            continue
        overlap_start = max(timestamp, steady_start)
        if interval_end <= overlap_start:
            continue
        max_steady = max(max_steady, current)
        if current >= target:
            seconds_at_target += interval_end - overlap_start

    steady_duration = max(0.0, benchmark_end - steady_start)
    target_coverage = (
        seconds_at_target / steady_duration if steady_duration > 0 else 0.0
    )
    return {
        "target_concurrent_sessions": target,
        "rampup_seconds": rampup_seconds,
        "max_observed_concurrency": max_observed,
        "max_observed_steady_state_concurrency": max_steady,
        "steady_state_duration_seconds": steady_duration,
        "seconds_at_or_above_target": seconds_at_target,
        "steady_state_target_coverage": target_coverage,
        "target_achieved": seconds_at_target > 0.0,
    }


def _validate_completed_child(child: CompiledDatasetRun) -> dict[str, Any]:
    rows = _read_request_metric_rows(child.config.output_dir)
    observed_session_count, intervals = _session_intervals(rows)
    issues: list[str] = []

    if not rows:
        issues.append("request_metrics_missing_or_empty")
    if observed_session_count is None:
        issues.append("session_id_missing_from_request_metrics")

    expected = child.expected_session_count
    dataset_complete: bool | None = None
    if expected is not None:
        dataset_complete = observed_session_count == expected
        if not dataset_complete:
            issues.append("dataset_session_count_mismatch")

    load_validation: dict[str, Any] | None = None
    if child.load_point is not None:
        traffic = child.config.traffic_scheduler
        load_validation = _measure_achieved_concurrency(
            intervals,
            target=child.load_point.target_concurrent_sessions,
            rampup_seconds=float(traffic.rampup_seconds),  # type: ignore[attr-defined]
        )
        if not load_validation["target_achieved"]:
            issues.append("configured_concurrency_not_achieved_in_steady_state")

    return {
        "valid": not issues,
        "issues": issues,
        "expected_session_count": expected,
        "observed_session_count": observed_session_count,
        "dataset_complete": dataset_complete,
        "load_validation": load_validation,
    }


def _child_contract_failure(
    child: CompiledDatasetRun,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    failure: dict[str, Any] = {
        "target_id": child.target_id,
        "dataset_id": child.dataset_id,
        "error_type": "BenchmarkContractError",
        "error": "Post-run benchmark contract failed: "
        + ", ".join(str(issue) for issue in validation.get("issues", [])),
        "run_dir": str(child.config.output_dir),
        "execution_validation": dict(validation),
    }
    if child.load_point is not None:
        failure["load_point_id"] = child.load_point.id
        failure["load"] = child.load_point.to_mapping()
    return failure


def _run_child(
    child: CompiledDatasetRun,
    completed: list[CompletedDatasetRun],
    failures: list[dict[str, Any]],
    *,
    endpoint: Any | None = None,
) -> None:
    try:
        if endpoint is None:
            result = manage_benchmark_run(child.config)
        else:
            result = run_benchmark_with_endpoint(child.config, endpoint)
        execution_validation = _validate_completed_child(child)
        if not execution_validation["valid"]:
            failure = _child_contract_failure(child, execution_validation)
            logger.error(
                "Named benchmark child produced an invalid measurement: %s (%s)",
                _failure_label(failure),
                failure["error"],
            )
            failures.append(failure)
            return
        completed.append(
            CompletedDatasetRun(
                target_id=child.target_id,
                dataset_id=child.dataset_id,
                run_dir=child.config.output_dir,
                metrics=result.metrics,
                load_point_id=(
                    None if child.load_point is None else child.load_point.id
                ),
                load=_load_point_payload(child.load_point),
                execution_validation=execution_validation,
            )
        )
    except Exception as exc:
        logger.exception(
            "Named benchmark child failed: %s",
            _failure_label(
                {
                    "target_id": child.target_id,
                    "load_point_id": (
                        None if child.load_point is None else child.load_point.id
                    ),
                    "dataset_id": child.dataset_id,
                }
            ),
        )
        failures.append(_child_failure(child, exc))


def _write_results(
    plan: NamedBenchmarkPlan,
    completed: Sequence[CompletedDatasetRun],
    failures: Sequence[Mapping[str, Any]],
) -> None:
    expected_target_ids = [record["target_id"] for record in plan.targets]
    load = plan.benchmark.execution.load
    if load is None:
        summary = build_named_benchmark_results(
            plan.benchmark,
            completed,
            expected_target_ids=expected_target_ids,
        )
    else:
        summary = build_named_benchmark_sweep_results(
            plan.benchmark,
            completed,
            load_points=load.points,
            expected_target_ids=expected_target_ids,
        )
    summary["run_failures"] = list(failures)
    _write_json(plan.parent_dir / "benchmark_summary.json", summary)

    rows: list[dict[str, Any]] = []
    for target in summary["targets"]:
        if load is None:
            for dataset in target["datasets"]:
                rows.append(
                    {
                        "benchmark_id": plan.benchmark.id,
                        "target_id": target["target_id"],
                        **dataset,
                    }
                )
            continue
        for load_result in target["load_points"]:
            for dataset in load_result["datasets"]:
                rows.append(
                    {
                        "benchmark_id": plan.benchmark.id,
                        "target_id": target["target_id"],
                        "load_point_id": load_result["load_point_id"],
                        "load": load_result["load"],
                        **dataset,
                    }
                )
    _write_jsonl(plan.parent_dir / "dataset_results.jsonl", rows)
    _write_jsonl(plan.parent_dir / "run_failures.jsonl", failures)
    _write_json(
        plan.parent_dir / "run_status.json",
        {
            "status": "failed" if failures else "complete",
            "compiled_runs": len(plan.children),
            "completed_runs": len(completed),
            "failed_runs": len(failures),
        },
    )


def _target_binding(target: BenchmarkConfig) -> dict[str, Any]:
    target_dict = to_serializable_config_dict(target)
    binding = {
        key: copy.deepcopy(target_dict.get(key))
        for key in ("client", "server", "endpoint")
        if target_dict.get(key) is not None
    }
    redacted = redact_config_secrets(binding)
    assert isinstance(redacted, dict)
    return redacted


def _deduplicate_target_configs(
    targets: Sequence[BenchmarkConfig],
) -> tuple[list[BenchmarkConfig], list[dict[str, Any]]]:
    """Drop expansions caused only by target YAML workload fields.

    ``target_config`` is parsed as an ordinary ``BenchmarkConfig``, so a
    concurrency/session/runtime ``!expand`` may produce many resolved configs
    even though the named layer intentionally keeps only client, server, and
    endpoint bindings.  Deduplicating that projection prevents ignored
    workload sweeps from multiplying an otherwise identical target.
    """

    unique_targets: list[BenchmarkConfig] = []
    unique_bindings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for target in targets:
        binding = _target_binding(target)
        identity = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        if identity in seen:
            continue
        seen.add(identity)
        unique_targets.append(target)
        unique_bindings.append(binding)
    return unique_targets, unique_bindings


def _unique_target_ids(bindings: Sequence[Mapping[str, Any]]) -> list[str]:
    target_ids: list[str] = []
    seen: set[str] = set()
    for binding in bindings:
        client = binding.get("client") or {}
        provider = str(client.get("provider") or client.get("type") or "target")
        model = str(client.get("model") or "model")
        digest = hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:8]
        target_id = f"{_slug(provider)}-{_slug(model)}-{digest}"
        if target_id in seen:
            raise ValueError(
                "target_config expanded to duplicate target bindings: " f"{target_id}"
            )
        seen.add(target_id)
        target_ids.append(target_id)
    return target_ids


def _create_parent_dir(
    config: NamedBenchmarkConfig,
    benchmark: Benchmark,
    target_bindings: Sequence[Mapping[str, Any]],
) -> Path:
    identity = {
        "benchmark": benchmark.to_mapping(),
        "targets": target_bindings,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = Path(config.output_dir) / benchmark.id
    parent = base / f"{timestamp}-{digest}"
    suffix = 1
    while parent.exists():
        suffix += 1
        parent = base / f"{timestamp}-{digest}-{suffix}"
    parent.mkdir(parents=True)
    return parent


def _write_plan_manifest(
    plan: NamedBenchmarkPlan,
    config: NamedBenchmarkConfig,
    *,
    runner_state: Mapping[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Retain the original field for consumers of the initial schema while
        # recording whether that commit completely describes the executable
        # working tree in the structured state below.
        "runner_revision": runner_state["git_commit"],
        "runner_state": dict(runner_state),
        "benchmark_reference": config.benchmark,
        "target_config": str(Path(config.target_config).resolve()),
        "dataset_root": (
            str(Path(config.dataset_root).resolve()) if config.dataset_root else None
        ),
        "dry_run": config.dry_run,
        "benchmark": plan.benchmark.to_mapping(),
        "targets": list(plan.targets),
        "children": [],
    }
    load = plan.benchmark.execution.load
    if load is not None:
        payload["load_points"] = [
            {"load_point_id": point.id, "load": point.to_mapping()}
            for point in load.points
        ]
    for child in plan.children:
        child_record: dict[str, Any] = {
            "target_id": child.target_id,
            "dataset_id": child.dataset_id,
            "compiled_config": str(child.config_path.relative_to(plan.parent_dir)),
            "expected_session_count": child.expected_session_count,
        }
        if child.load_point is not None:
            child_record["load_point_id"] = child.load_point.id
            child_record["load"] = child.load_point.to_mapping()
        payload["children"].append(child_record)
    _write_json(plan.parent_dir / "benchmark_manifest.json", payload)


def _validate_materialized_dataset_provenance(
    benchmark: Benchmark,
    *,
    dataset_root: str,
) -> None:
    """Reject noncanonical or modified Hugging Face ASR trace materializations.

    ``prepare_hf_audio_trace.py`` writes one ``preparation.json`` beside each
    generated ``manifest.jsonl``.  A named canonical run is allowed to consume
    that trace only when the catalog source selection, preparation record, and
    actual manifest all agree.  Fixture and non-Hugging-Face sources are left
    to their own materializers rather than being guessed at here.
    """

    if benchmark.modality is not Modality.ASR:
        return

    for dataset in benchmark.datasets:
        if dataset.source.kind.casefold() != "huggingface":
            continue
        _validate_huggingface_asr_trace(dataset, dataset_root=dataset_root)


def _validate_huggingface_asr_trace(
    dataset: DatasetCase,
    *,
    dataset_root: str,
) -> None:
    context = f"dataset={dataset.id}"
    if dataset.source.config is None:
        raise ValueError(
            f"{context}: canonical Hugging Face ASR sources require source.config"
        )
    if dataset.source.expected_rows is None:
        raise ValueError(
            f"{context}: canonical Hugging Face ASR sources require "
            "source.expected_rows"
        )

    session_generator = _expand_dataset_root(
        dataset.session_generator,
        dataset_root,
    )
    if session_generator.get("type") != "trace":
        raise ValueError(
            f"{context}: materialized Hugging Face ASR data requires "
            "session_generator.type=trace"
        )
    trace_file = session_generator.get("trace_file")
    if not isinstance(trace_file, str) or not trace_file.strip():
        raise ValueError(f"{context}: session_generator.trace_file is required")

    manifest_path = Path(trace_file).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"{context}: trace manifest not found: {manifest_path}")
    preparation_path = manifest_path.parent / _PREPARATION_METADATA_NAME
    metadata = _read_json_mapping(
        preparation_path,
        context=f"{context} preparation metadata",
    )

    if metadata.get("schema_version") != 1:
        raise ValueError(
            f"{context}: preparation schema_version must be 1, got "
            f"{metadata.get('schema_version')!r}"
        )
    if metadata.get("canonical") is not True:
        raise ValueError(
            f"{context}: preparation is noncanonical; do not publish a trace "
            "created with --max-samples under a canonical benchmark ID"
        )

    source = _require_mapping(metadata.get("source"), context=f"{context}.source")
    expected_source = {
        "repo_id": dataset.source.uri,
        "revision": dataset.source.revision,
        "config": dataset.source.config,
        "split": dataset.source.split,
    }
    for field, expected in expected_source.items():
        actual = source.get(field)
        if actual != expected:
            raise ValueError(
                f"{context}: preparation source.{field}={actual!r} does not "
                f"match catalog source.{field}={expected!r}"
            )

    selection = _require_mapping(
        metadata.get("selection"),
        context=f"{context}.selection",
    )
    if selection.get("order") != "source_order":
        raise ValueError(
            f"{context}: canonical preparation selection.order must be "
            f"'source_order', got {selection.get('order')!r}"
        )
    if selection.get("max_samples") is not None:
        raise ValueError(
            f"{context}: canonical preparation must have max_samples=null; "
            f"got {selection.get('max_samples')!r}"
        )

    expected_rows = dataset.source.expected_rows
    for field in ("total_rows", "materialized_rows"):
        actual = selection.get(field)
        if type(actual) is not int or actual != expected_rows:
            raise ValueError(
                f"{context}: preparation selection.{field}={actual!r} does not "
                f"match catalog expected_rows={expected_rows}"
            )

    manifest = _require_mapping(
        metadata.get("manifest"),
        context=f"{context}.manifest",
    )
    metadata_manifest_path = manifest.get("path")
    if not isinstance(metadata_manifest_path, str) or not metadata_manifest_path:
        raise ValueError(f"{context}: preparation manifest.path is required")
    relative_manifest_path = Path(metadata_manifest_path)
    if relative_manifest_path.is_absolute():
        raise ValueError(
            f"{context}: preparation manifest.path must be relative, got "
            f"{metadata_manifest_path!r}"
        )
    prepared_manifest_path = (
        preparation_path.parent / relative_manifest_path
    ).resolve()
    if prepared_manifest_path != manifest_path:
        raise ValueError(
            f"{context}: preparation manifest.path resolves to "
            f"{prepared_manifest_path}, not configured trace {manifest_path}"
        )

    expected_metadata_sha256 = _parse_sha256(
        manifest.get("sha256"),
        context=f"{context}.manifest.sha256",
    )
    actual_sha256 = _sha256_file(manifest_path)
    if actual_sha256 != expected_metadata_sha256:
        raise ValueError(
            f"{context}: trace manifest checksum mismatch; expected "
            f"{expected_metadata_sha256}, got {actual_sha256}. The trace was "
            "modified after preparation."
        )

    if dataset.source.checksum is not None:
        catalog_sha256 = _parse_sha256(
            dataset.source.checksum,
            context=f"{context}.source.checksum",
        )
        if actual_sha256 != catalog_sha256:
            raise ValueError(
                f"{context}: trace manifest checksum {actual_sha256} does not "
                f"match catalog source.checksum {catalog_sha256}"
            )

    actual_rows = _validate_manifest_rows(
        manifest_path,
        expected_revision=dataset.source.revision,
        context=context,
    )
    if actual_rows != expected_rows:
        raise ValueError(
            f"{context}: trace manifest contains {actual_rows} rows, expected "
            f"{expected_rows}"
        )


def _read_json_mapping(path: Path, *, context: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"{context} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not valid UTF-8 JSON: {path}") from exc
    return _require_mapping(value, context=context)


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _parse_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a SHA-256 string")
    match = _SHA256_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            f"{context} must be 64 hexadecimal characters, optionally "
            "prefixed by 'sha256:'"
        )
    return match.group(1).lower()


def _validate_manifest_rows(
    path: Path,
    *,
    expected_revision: str,
    context: str,
) -> int:
    row_count = 0
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    raise ValueError(
                        f"{context}: trace manifest contains a blank row at "
                        f"line {line_number}"
                    )
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{context}: trace manifest line {line_number} is not "
                        "valid JSON"
                    ) from exc
                row = _require_mapping(
                    row,
                    context=f"{context} trace manifest line {line_number}",
                )
                if row.get("source_revision") != expected_revision:
                    raise ValueError(
                        f"{context}: trace manifest line {line_number} has "
                        f"source_revision={row.get('source_revision')!r}, expected "
                        f"{expected_revision!r}"
                    )
                if row.get("source_row_index") != row_count:
                    raise ValueError(
                        f"{context}: trace manifest line {line_number} has "
                        f"source_row_index={row.get('source_row_index')!r}, "
                        f"expected {row_count}"
                    )
                row_count += 1
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{context}: unable to read trace manifest {path}") from exc
    return row_count


def _expand_dataset_root(value: Any, dataset_root: str) -> Any:
    if isinstance(value, str):
        if _DATASET_ROOT_PLACEHOLDER not in value:
            return value
        if not dataset_root:
            raise ValueError(
                f"Benchmark contains {_DATASET_ROOT_PLACEHOLDER} but "
                "NamedBenchmarkConfig.dataset_root is empty."
            )
        return value.replace(_DATASET_ROOT_PLACEHOLDER, dataset_root)
    if isinstance(value, dict):
        return {
            key: _expand_dataset_root(item, dataset_root) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_expand_dataset_root(item, dataset_root) for item in value]
    return value


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.casefold()).strip("-")
    return slug or "target"


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _runner_state(target_config_path: Path) -> dict[str, Any]:
    """Fingerprint the executable checkout without persisting source content."""

    target_config_sha256 = _sha256_file(target_config_path.resolve())
    try:
        repository = _git(
            "rev-parse",
            "--show-toplevel",
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).stdout.strip()
        repository_root = Path(repository).resolve()
        git_commit = _git(
            "rev-parse",
            "HEAD",
            cwd=repository_root,
            text=True,
        ).stdout.strip()
        status = _git(
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            cwd=repository_root,
        ).stdout
        tracked_diff = _git(
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
            cwd=repository_root,
        ).stdout
        untracked = _git(
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            cwd=repository_root,
        ).stdout
    except (OSError, subprocess.CalledProcessError, UnicodeError):
        return {
            "git_commit": None,
            "dirty": None,
            "working_tree_diff_sha256": None,
            "target_config_sha256": target_config_sha256,
        }

    digest = hashlib.sha256()
    _hash_labeled_bytes(digest, b"tracked_diff", tracked_diff)
    for relative_path_bytes in sorted(path for path in untracked.split(b"\0") if path):
        _hash_labeled_bytes(digest, b"untracked_path", relative_path_bytes)
        relative_path = relative_path_bytes.decode("utf-8", errors="surrogateescape")
        path = repository_root / relative_path
        if path.is_symlink():
            target = os.readlink(path)
            _hash_labeled_bytes(
                digest,
                b"symlink_target",
                os.fsencode(target),
            )
        elif path.is_file():
            _hash_labeled_bytes(
                digest,
                b"untracked_file_sha256",
                bytes.fromhex(_sha256_file(path)),
            )
        else:
            # A file can disappear between git discovery and hashing. Preserve
            # that state deterministically instead of silently omitting it.
            _hash_labeled_bytes(digest, b"missing_untracked_file", b"")

    return {
        "git_commit": git_commit or None,
        "dirty": bool(status),
        "working_tree_diff_sha256": digest.hexdigest(),
        "target_config_sha256": target_config_sha256,
    }


def _git(
    *args: str,
    cwd: Path,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=text,
    )


def _hash_labeled_bytes(
    digest: Any,
    label: bytes,
    value: bytes,
) -> None:
    digest.update(len(label).to_bytes(8, "big"))
    digest.update(label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Unable to read file for SHA-256: {path}") from exc
    return digest.hexdigest()
