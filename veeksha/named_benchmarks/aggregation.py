"""Dataset-aware aggregation for named benchmark results.

Named benchmarks execute one ordinary Veeksha run per target and dataset.  This
module keeps those child results intact and derives only the cross-dataset
values explicitly requested by the benchmark manifest.  In particular,
percentiles are always recomputed from request-level observations; percentile
summaries from child runs are never averaged.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from statistics import fmean
from typing import Any

_MISSING = object()
_PERCENTILE_PATH = re.compile(
    r"(?:^|[^a-z0-9])(?:p(?:ct|ercentile)?\s*\d+(?:[._]\d+)?|"
    r"percentile|quantile|median)"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CompletedDatasetRun:
    """A completed Veeksha child run belonging to one named benchmark."""

    target_id: str
    dataset_id: str
    run_dir: str | Path
    metrics: Any


def build_named_benchmark_results(
    benchmark_spec: Any,
    child_runs: Sequence[CompletedDatasetRun | Mapping[str, Any] | Any],
    *,
    expected_target_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build deterministic dataset- and benchmark-level result data.

    ``benchmark_spec`` is intentionally duck-typed so this helper can consume a
    parsed schema object or the equivalent mapping.  It must expose ``id`` (or
    ``benchmark_id``), ``metrics``, and, when missing-dataset diagnostics are
    desired, ``datasets``.  Metric declarations use the following fields:

    - ``id``, ``role``, and ``unit``;
    - ``dataset_aggregation`` and ``benchmark_aggregation``;
    - each aggregation declares ``method`` and the operands required by that
      method: ``source``; ``source`` plus ``quantiles``; or exact ``numerator``
      and ``denominator`` paths.

    Supported dataset methods are ``scalar``, ``distribution``,
    ``ratio_of_sums``, and ``none``.  Supported benchmark methods are
    ``equal_dataset_mean``, ``pooled_distribution``, ``ratio_of_sums``, and
    ``none``.  ``scalar`` and ``distribution`` are also accepted at benchmark
    level only for a single-dataset benchmark.
    """

    benchmark_id = str(
        _required_value(benchmark_spec, ("id", "benchmark_id"), "benchmark id")
    )
    declarations = list(_value(benchmark_spec, "metrics", default=()) or ())
    normalized_children = [_normalize_child_run(child) for child in child_runs]
    normalized_children.sort(
        key=lambda child: (child.target_id, child.dataset_id, str(child.run_dir))
    )
    _reject_duplicate_children(normalized_children)

    expected_dataset_ids = _expected_dataset_ids(benchmark_spec)
    targets: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []

    children_by_target: dict[str, list[_NormalizedChildRun]] = {}
    for child in normalized_children:
        children_by_target.setdefault(child.target_id, []).append(child)
    for target_id in expected_target_ids:
        children_by_target.setdefault(str(target_id), [])

    if not normalized_children:
        all_diagnostics.append(
            {
                "level": "benchmark",
                "benchmark_id": benchmark_id,
                "reason": "no_child_runs",
            }
        )

    for target_id in sorted(children_by_target):
        children = children_by_target[target_id]
        dataset_results: list[dict[str, Any]] = []
        target_diagnostics: list[dict[str, Any]] = []

        for child in children:
            dataset_result, diagnostics = _resolve_dataset_result(
                benchmark_id, child, declarations
            )
            dataset_results.append(dataset_result)
            target_diagnostics.extend(diagnostics)

        present_dataset_ids = {child.dataset_id for child in children}
        for dataset_id in sorted(expected_dataset_ids - present_dataset_ids):
            target_diagnostics.append(
                {
                    "level": "dataset",
                    "benchmark_id": benchmark_id,
                    "target_id": target_id,
                    "dataset_id": dataset_id,
                    "reason": "missing_dataset_run",
                }
            )

        benchmark_metrics, benchmark_diagnostics = _resolve_benchmark_metrics(
            benchmark_id=benchmark_id,
            target_id=target_id,
            children=children,
            declarations=declarations,
            expected_dataset_ids=expected_dataset_ids,
        )
        target_diagnostics.extend(benchmark_diagnostics)
        target_diagnostics = _sorted_diagnostics(target_diagnostics)
        all_diagnostics.extend(target_diagnostics)

        targets.append(
            {
                "target_id": target_id,
                "sample_count": sum(child.sample_count for child in children),
                "dataset_count": len(children),
                "expected_dataset_count": (
                    len(expected_dataset_ids) if expected_dataset_ids else len(children)
                ),
                "datasets": dataset_results,
                "benchmark_metrics": benchmark_metrics,
                "missing_metric_diagnostics": target_diagnostics,
            }
        )

    result = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "targets": targets,
        "missing_metric_diagnostics": _sorted_diagnostics(all_diagnostics),
    }
    return _json_safe(result)


def write_named_benchmark_results(
    path: str | Path,
    benchmark_spec: Any,
    child_runs: Sequence[CompletedDatasetRun | Mapping[str, Any] | Any],
) -> Path:
    """Build and atomically write a named benchmark result JSON document."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_named_benchmark_results(benchmark_spec, child_runs)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    temporary.replace(destination)
    return destination


@dataclass(frozen=True)
class _NormalizedChildRun:
    target_id: str
    dataset_id: str
    run_dir: Path
    metrics: Mapping[str, Any]
    request_rows: tuple[Mapping[str, Any], ...]
    sample_count: int


def _normalize_child_run(child: Any) -> _NormalizedChildRun:
    target_id = str(_required_value(child, ("target_id",), "child target_id"))
    dataset_id = str(_required_value(child, ("dataset_id",), "child dataset_id"))
    run_dir = Path(_required_value(child, ("run_dir",), "child run_dir"))
    metrics_value = _required_value(
        child, ("metrics", "evaluation_result"), "child metrics"
    )
    if not isinstance(metrics_value, Mapping):
        metrics_value = _value(metrics_value, "metrics", default=_MISSING)
    if not isinstance(metrics_value, Mapping):
        raise TypeError(
            f"Metrics for target={target_id!r}, dataset={dataset_id!r} must be a mapping "
            "or an EvaluationResult-like object exposing .metrics."
        )

    rows = tuple(_load_request_rows(run_dir))
    sample_count = len(rows) if rows else _sample_count_from_metrics(metrics_value)
    return _NormalizedChildRun(
        target_id=target_id,
        dataset_id=dataset_id,
        run_dir=run_dir,
        metrics=metrics_value,
        request_rows=rows,
        sample_count=sample_count,
    )


def _reject_duplicate_children(children: Sequence[_NormalizedChildRun]) -> None:
    seen: set[tuple[str, str]] = set()
    for child in children:
        key = (child.target_id, child.dataset_id)
        if key in seen:
            raise ValueError(
                "A named benchmark result accepts one completed run per target and "
                f"dataset; duplicate found for target={key[0]!r}, dataset={key[1]!r}."
            )
        seen.add(key)


def _resolve_dataset_result(
    benchmark_id: str,
    child: _NormalizedChildRun,
    declarations: Sequence[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for declaration in declarations:
        metric_id = str(_required_value(declaration, ("id",), "metric id"))
        aggregation = _value(declaration, "dataset_aggregation", default=None)
        metric, reasons = _resolve_dataset_metric(child, declaration, aggregation)
        if metric is not None:
            resolved[metric_id] = metric
        diagnostics.extend(
            _metric_diagnostics(
                reasons,
                level="dataset",
                benchmark_id=benchmark_id,
                target_id=child.target_id,
                dataset_id=child.dataset_id,
                metric_id=metric_id,
            )
        )

    diagnostics = _sorted_diagnostics(diagnostics)
    return (
        {
            "dataset_id": child.dataset_id,
            "run_dir": str(child.run_dir),
            "sample_count": child.sample_count,
            "resolved_metrics": resolved,
            "child_metrics": _json_safe(child.metrics),
            "missing_metric_diagnostics": diagnostics,
        },
        diagnostics,
    )


def _resolve_dataset_metric(
    child: _NormalizedChildRun, declaration: Any, aggregation: Any
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    method = _method(aggregation)
    metadata = _metric_metadata(declaration, method)

    if method == "none":
        return None, []
    failure_reason = _request_success_requirement_reason(declaration, (child,))
    if failure_reason is not None:
        return None, [failure_reason]
    if method == "scalar":
        source = str(_required_value(aggregation, ("source",), "scalar source"))
        value = _finite_float(_resolve_path(child.metrics, source))
        if value is None:
            return None, [{"reason": "missing_or_non_numeric_source", "path": source}]
        return {**metadata, "value": value}, []
    if method == "distribution":
        source = str(_required_value(aggregation, ("source",), "distribution source"))
        quantiles = _quantiles(aggregation)
        values, missing_rows = _request_values(child.request_rows, source)
        if not values:
            return None, [
                {
                    "reason": "missing_request_distribution",
                    "path": source,
                    "request_count": len(child.request_rows),
                }
            ]
        metric = {
            **metadata,
            "source": source,
            "request_count": len(child.request_rows),
            "observation_count": len(values),
            "missing_request_count": missing_rows,
            "quantiles": _compute_quantiles(values, quantiles),
        }
        reasons = []
        if missing_rows:
            reasons.append(
                {
                    "reason": "partially_missing_request_distribution",
                    "path": source,
                    "missing_request_count": missing_rows,
                    "request_count": len(child.request_rows),
                }
            )
        return metric, reasons
    if method == "ratio_of_sums":
        numerator_path, denominator_path = _ratio_paths(aggregation)
        scale = _ratio_scale(aggregation)
        numerator = _finite_float(_resolve_path(child.metrics, numerator_path))
        denominator = _finite_float(_resolve_path(child.metrics, denominator_path))
        reasons = _ratio_reasons(
            numerator, denominator, numerator_path, denominator_path
        )
        if reasons:
            return None, reasons
        assert numerator is not None and denominator is not None
        return {
            **metadata,
            "value": (numerator / denominator) * scale,
            "numerator_sum": numerator,
            "denominator_sum": denominator,
            "scale": scale,
        }, []

    raise ValueError(f"Unsupported dataset aggregation method: {method!r}")


def _resolve_benchmark_metrics(
    *,
    benchmark_id: str,
    target_id: str,
    children: Sequence[_NormalizedChildRun],
    declarations: Sequence[Any],
    expected_dataset_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []

    for declaration in declarations:
        metric_id = str(_required_value(declaration, ("id",), "metric id"))
        aggregation = _value(declaration, "benchmark_aggregation", default=None)
        metric, reasons = _resolve_benchmark_metric(
            children,
            declaration,
            aggregation,
            expected_dataset_ids=expected_dataset_ids,
        )
        if metric is not None:
            resolved[metric_id] = metric
        diagnostics.extend(
            _metric_diagnostics(
                reasons,
                level="benchmark",
                benchmark_id=benchmark_id,
                target_id=target_id,
                metric_id=metric_id,
            )
        )

    return resolved, _sorted_diagnostics(diagnostics)


def _resolve_benchmark_metric(
    children: Sequence[_NormalizedChildRun],
    declaration: Any,
    aggregation: Any,
    *,
    expected_dataset_ids: set[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    method = _method(aggregation)
    metadata = _metric_metadata(declaration, method)
    present_dataset_ids = {child.dataset_id for child in children}
    missing_run_ids = expected_dataset_ids - present_dataset_ids
    expected_dataset_count = (
        len(expected_dataset_ids) if expected_dataset_ids else len(children)
    )

    if method == "none":
        return None, []

    failure_reason = _request_success_requirement_reason(declaration, children)
    if failure_reason is not None:
        return None, [failure_reason]

    if method in {"equal_dataset_mean", "scalar"}:
        source = str(_required_value(aggregation, ("source",), f"{method} source"))
        if _looks_like_percentile_path(source):
            raise ValueError(
                f"Refusing to average percentile-like source {source!r}; use "
                "pooled_distribution so quantiles are recomputed from request rows."
            )
        if method == "scalar" and len(children) != 1:
            return None, [
                {
                    "reason": "benchmark_scalar_requires_one_dataset",
                    "path": source,
                    "dataset_count": len(children),
                }
            ]
        values: list[float] = []
        missing_datasets: set[str] = set(missing_run_ids)
        for child in children:
            value = _finite_float(_resolve_path(child.metrics, source))
            if value is None:
                missing_datasets.add(child.dataset_id)
            else:
                values.append(value)
        if not values:
            return None, [
                {
                    "reason": "missing_or_non_numeric_source_for_all_datasets",
                    "path": source,
                    "missing_dataset_ids": sorted(missing_datasets),
                }
            ]
        metric = {
            **metadata,
            "value": values[0] if method == "scalar" else fmean(values),
            "dataset_count": len(values),
            "expected_dataset_count": expected_dataset_count,
            "complete": not missing_datasets,
        }
        reasons = []
        if missing_datasets:
            reasons.append(
                {
                    "reason": "missing_source_for_some_datasets",
                    "path": source,
                    "missing_dataset_ids": sorted(missing_datasets),
                }
            )
        return metric, reasons

    if method in {"pooled_distribution", "distribution"}:
        source = str(_required_value(aggregation, ("source",), f"{method} source"))
        quantiles = _quantiles(aggregation)
        if method == "distribution" and len(children) != 1:
            return None, [
                {
                    "reason": "benchmark_distribution_requires_one_dataset",
                    "path": source,
                    "dataset_count": len(children),
                }
            ]
        values: list[float] = []
        request_count = 0
        missing_request_count = 0
        missing_datasets: set[str] = set(missing_run_ids)
        for child in children:
            child_values, child_missing = _request_values(child.request_rows, source)
            values.extend(child_values)
            request_count += len(child.request_rows)
            missing_request_count += child_missing
            if not child_values:
                missing_datasets.add(child.dataset_id)
        if not values:
            return None, [
                {
                    "reason": "missing_request_distribution_for_all_datasets",
                    "path": source,
                    "missing_dataset_ids": sorted(missing_datasets),
                }
            ]
        metric = {
            **metadata,
            "source": source,
            "request_count": request_count,
            "observation_count": len(values),
            "missing_request_count": missing_request_count,
            "dataset_count": sum(
                child.dataset_id not in missing_datasets for child in children
            ),
            "expected_dataset_count": expected_dataset_count,
            "complete": not missing_datasets and missing_request_count == 0,
            "quantiles": _compute_quantiles(values, quantiles),
        }
        reasons = []
        if missing_datasets or missing_request_count:
            reasons.append(
                {
                    "reason": "partially_missing_pooled_distribution",
                    "path": source,
                    "missing_dataset_ids": sorted(missing_datasets),
                    "missing_request_count": missing_request_count,
                    "request_count": request_count,
                }
            )
        return metric, reasons

    if method == "ratio_of_sums":
        numerator_path, denominator_path = _ratio_paths(aggregation)
        scale = _ratio_scale(aggregation)
        numerator_sum = 0.0
        denominator_sum = 0.0
        included_datasets: list[str] = []
        missing_datasets: set[str] = set(missing_run_ids)
        zero_denominator_datasets: list[str] = []

        for child in children:
            numerator = _finite_float(_resolve_path(child.metrics, numerator_path))
            denominator = _finite_float(_resolve_path(child.metrics, denominator_path))
            if numerator is None or denominator is None:
                missing_datasets.add(child.dataset_id)
                continue
            if denominator < 0:
                missing_datasets.add(child.dataset_id)
                continue
            if denominator == 0:
                zero_denominator_datasets.append(child.dataset_id)
                continue
            numerator_sum += numerator
            denominator_sum += denominator
            included_datasets.append(child.dataset_id)

        if not included_datasets or denominator_sum == 0:
            return None, [
                {
                    "reason": "ratio_has_no_valid_datasets",
                    "numerator_path": numerator_path,
                    "denominator_path": denominator_path,
                    "missing_dataset_ids": sorted(missing_datasets),
                    "zero_denominator_dataset_ids": sorted(zero_denominator_datasets),
                }
            ]

        metric = {
            **metadata,
            "value": (numerator_sum / denominator_sum) * scale,
            "numerator_sum": numerator_sum,
            "denominator_sum": denominator_sum,
            "scale": scale,
            "dataset_count": len(included_datasets),
            "expected_dataset_count": expected_dataset_count,
            "complete": not missing_datasets and not zero_denominator_datasets,
        }
        reasons = []
        if missing_datasets or zero_denominator_datasets:
            reasons.append(
                {
                    "reason": "ratio_missing_some_datasets",
                    "numerator_path": numerator_path,
                    "denominator_path": denominator_path,
                    "missing_dataset_ids": sorted(missing_datasets),
                    "zero_denominator_dataset_ids": sorted(zero_denominator_datasets),
                }
            )
        return metric, reasons

    raise ValueError(f"Unsupported benchmark aggregation method: {method!r}")


def _load_request_rows(run_dir: Path) -> Iterable[Mapping[str, Any]]:
    path = run_dir / "metrics" / "request_level_metrics.jsonl"
    if not path.is_file():
        return ()
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"Expected a JSON object in {path} at line {line_number}."
                )
            rows.append(row)
    return rows


def _request_values(
    rows: Sequence[Mapping[str, Any]], source: str
) -> tuple[list[float], int]:
    values: list[float] = []
    missing_rows = 0
    for row in rows:
        raw = _resolve_path(row, source)
        row_values = list(_flatten_finite_numbers(raw))
        if not row_values:
            missing_rows += 1
        else:
            values.extend(row_values)
    return values, missing_rows


def _flatten_finite_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_finite_numbers(item)
        return
    number = _finite_float(value)
    if number is not None:
        yield number


def _compute_quantiles(
    values: Sequence[float], quantiles: Sequence[float]
) -> dict[str, float]:
    ordered = sorted(values)
    result: dict[str, float] = {}
    for quantile in quantiles:
        if len(ordered) == 1:
            value = ordered[0]
        else:
            rank = quantile * (len(ordered) - 1)
            lower_index = math.floor(rank)
            upper_index = math.ceil(rank)
            if lower_index == upper_index:
                value = ordered[lower_index]
            else:
                fraction = rank - lower_index
                value = ordered[lower_index] + fraction * (
                    ordered[upper_index] - ordered[lower_index]
                )
        result[_quantile_label(quantile)] = value
    return result


def _quantiles(aggregation: Any) -> tuple[float, ...]:
    raw = _required_value(
        aggregation, ("quantiles", "percentiles"), "distribution quantiles"
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("Distribution quantiles must be a sequence of numbers.")
    values = sorted({_required_quantile(item) for item in raw})
    if not values:
        raise ValueError("Distribution quantiles must not be empty.")
    return tuple(values)


def _required_quantile(value: Any) -> float:
    quantile = _finite_float(value)
    if quantile is None or not 0.0 < quantile < 1.0:
        raise ValueError(
            f"Invalid quantile {value!r}; expected a value between 0 and 1."
        )
    return quantile


def _quantile_label(quantile: float) -> str:
    percentile = quantile * 100.0
    if percentile.is_integer():
        return f"p{int(percentile)}"
    return f"p{format(percentile, 'g')}"


def _ratio_paths(aggregation: Any) -> tuple[str, str]:
    numerator = str(
        _required_value(aggregation, ("numerator",), "ratio numerator path")
    )
    denominator = str(
        _required_value(aggregation, ("denominator",), "ratio denominator path")
    )
    return numerator, denominator


def _ratio_scale(aggregation: Any) -> float:
    scale = _finite_float(_value(aggregation, "scale", default=1.0))
    if scale is None or scale <= 0:
        raise ValueError("ratio_of_sums scale must be a positive finite number")
    return scale


def _ratio_reasons(
    numerator: float | None,
    denominator: float | None,
    numerator_path: str,
    denominator_path: str,
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    if numerator is None:
        reasons.append(
            {"reason": "missing_or_non_numeric_numerator", "path": numerator_path}
        )
    if denominator is None:
        reasons.append(
            {
                "reason": "missing_or_non_numeric_denominator",
                "path": denominator_path,
            }
        )
    elif denominator == 0:
        reasons.append({"reason": "zero_denominator", "path": denominator_path})
    elif denominator < 0:
        reasons.append({"reason": "negative_denominator", "path": denominator_path})
    return reasons


def _method(aggregation: Any) -> str:
    if aggregation is None:
        return "none"
    raw = _value(aggregation, "method", default=aggregation)
    if isinstance(raw, Enum):
        raw = raw.value
    return str(raw).strip().lower()


def _metric_metadata(declaration: Any, method: str) -> dict[str, Any]:
    return {
        "role": str(_enum_value(_value(declaration, "role", default="diagnostic"))),
        "unit": str(_enum_value(_value(declaration, "unit", default=""))),
        "aggregation": method,
    }


def _metric_diagnostics(
    reasons: Sequence[Mapping[str, Any]],
    *,
    level: str,
    benchmark_id: str,
    target_id: str,
    metric_id: str,
    dataset_id: str | None = None,
) -> list[dict[str, Any]]:
    prefix: dict[str, Any] = {
        "level": level,
        "benchmark_id": benchmark_id,
        "target_id": target_id,
        "metric_id": metric_id,
    }
    if dataset_id is not None:
        prefix["dataset_id"] = dataset_id
    return [{**prefix, **dict(reason)} for reason in reasons]


def _expected_dataset_ids(benchmark_spec: Any) -> set[str]:
    datasets = _value(benchmark_spec, "datasets", default=()) or ()
    result: set[str] = set()
    for dataset in datasets:
        dataset_id = _value(dataset, "id", default=_MISSING)
        if dataset_id is _MISSING:
            dataset_id = _value(dataset, "dataset_id", default=_MISSING)
        if dataset_id is not _MISSING:
            result.add(str(dataset_id))
    return result


def _sample_count_from_metrics(metrics: Mapping[str, Any]) -> int:
    candidates = (
        "Number of Requests",
        "number_of_requests",
        "num_requests",
        "num_completed_requests",
        "completed_requests",
        "asr_final_sample_count",
        "sample_count",
    )
    for path in candidates:
        value = _finite_float(_resolve_path(metrics, path))
        if value is not None and value >= 0:
            return int(value)
    return 0


def _request_success_requirement_reason(
    declaration: Any,
    children: Sequence[_NormalizedChildRun],
) -> dict[str, Any] | None:
    """Withhold correctness metrics when unscored requests would bias them."""

    if not bool(_value(declaration, "requires_all_requests_successful", default=False)):
        return None

    incomplete_datasets: dict[str, dict[str, int]] = {}
    unknown_datasets: list[str] = []
    for child in children:
        counts = _request_lifecycle_counts(child.metrics)
        if counts is None:
            unknown_datasets.append(child.dataset_id)
        elif (
            counts["completed"] != counts["total"]
            or counts["errored"] != 0
            or counts["cancelled"] != 0
        ):
            incomplete_datasets[child.dataset_id] = counts

    if incomplete_datasets:
        return {
            "reason": "metric_invalid_due_to_unsuccessful_requests",
            "request_counts_by_dataset": dict(sorted(incomplete_datasets.items())),
        }
    if unknown_datasets:
        return {
            "reason": "request_lifecycle_counts_unavailable",
            "missing_dataset_ids": sorted(unknown_datasets),
        }
    return None


def _request_lifecycle_counts(metrics: Mapping[str, Any]) -> dict[str, int] | None:
    paths = {
        "total": "Number of Requests",
        "completed": "Number of Completed Requests",
        "errored": "Number of Errored Requests",
        "cancelled": "Number of Cancelled Requests",
    }
    counts: dict[str, int] = {}
    for name, path in paths.items():
        value = _finite_float(_resolve_path(metrics, path))
        if value is None or value < 0 or not value.is_integer():
            return None
        counts[name] = int(value)
    return counts


def _resolve_path(value: Any, path: str) -> Any:
    if not path:
        return _MISSING
    if isinstance(value, Mapping) and path in value:
        return value[path]
    current = value
    for segment in path.split("."):
        if isinstance(current, Mapping) and segment in current:
            current = current[segment]
        else:
            return _MISSING
    return current


def _required_value(value: Any, names: Sequence[str], label: str) -> Any:
    for name in names:
        found = _value(value, name, default=_MISSING)
        if found is not _MISSING and found is not None:
            return found
    raise ValueError(f"Missing required {label}; expected one of {tuple(names)!r}.")


def _value(value: Any, name: str, *, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_float(value: Any) -> float | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _looks_like_percentile_path(path: str) -> bool:
    return bool(_PERCENTILE_PATH.search(path))


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _sorted_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    safe = [_json_safe(item) for item in diagnostics]
    return sorted(
        safe,
        key=lambda item: (
            str(item.get("target_id", "")),
            str(item.get("dataset_id", "")),
            str(item.get("metric_id", "")),
            str(item.get("reason", "")),
            str(item.get("path", "")),
        ),
    )


def _json_safe(value: Any) -> Any:
    if value is _MISSING:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=repr)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


__all__ = [
    "CompletedDatasetRun",
    "build_named_benchmark_results",
    "write_named_benchmark_results",
]
