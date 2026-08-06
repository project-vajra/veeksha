"""Strict, target-independent schema for named Veeksha benchmarks."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

SCHEMA_VERSION = 1

_EnumT = TypeVar("_EnumT", bound=StrEnum)

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_FORBIDDEN_EXECUTION_FIELDS = frozenset(
    {"client", "server", "endpoint", "output_dir", "session_generator"}
)
_FLOATING_REVISIONS = frozenset({"head", "latest", "main", "master"})
_TARGET_CLIENT_FIELDS = frozenset(
    {
        "type",
        "provider",
        "model",
        "api_base",
        "api_key",
        "api_key_env",
        "voice_id",
        "language",
        "language_mode",
        "supported_languages",
    }
)


class BenchmarkSchemaError(ValueError):
    """Raised when a named benchmark manifest is invalid."""


class Modality(StrEnum):
    ASR = "asr"
    TTS = "tts"


class InteractionMode(StrEnum):
    STATIC = "static"
    STREAMING = "streaming"


class MetricRole(StrEnum):
    PRIMARY = "primary"
    DIAGNOSTIC = "diagnostic"


class AggregationMethod(StrEnum):
    """Supported reducers with well-defined cross-dataset semantics."""

    SCALAR = "scalar"
    DISTRIBUTION = "distribution"
    EQUAL_DATASET_MEAN = "equal_dataset_mean"
    POOLED_DISTRIBUTION = "pooled_distribution"
    RATIO_OF_SUMS = "ratio_of_sums"
    NONE = "none"


class LoadType(StrEnum):
    """Supported high-level load dimensions for a named benchmark."""

    CONCURRENCY_SWEEP = "concurrency_sweep"


@dataclass(frozen=True, slots=True)
class ConcurrencyLoadPoint:
    """One concrete concurrency value compiled into an ordinary run."""

    id: str
    target_concurrent_sessions: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "type": "concurrency",
            "target_concurrent_sessions": self.target_concurrent_sessions,
        }


@dataclass(frozen=True, slots=True)
class ConcurrencySweep:
    """A fixed, canonical sweep over concurrent live sessions."""

    type: LoadType
    values: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: object, *, context: str) -> ConcurrencySweep:
        data = _mapping(value, context=context)
        _reject_unknown(data, {"type", "values"}, context=context)
        load_type = _enum_value(
            LoadType,
            data.get("type"),
            context=f"{context}.type",
        )
        values = _positive_sorted_unique_ints(
            data.get("values"),
            context=f"{context}.values",
        )
        return cls(type=load_type, values=values)

    @property
    def points(self) -> tuple[ConcurrencyLoadPoint, ...]:
        return tuple(
            ConcurrencyLoadPoint(
                id=f"concurrency-{value:04d}",
                target_concurrent_sessions=value,
            )
            for value in self.values
        )

    def to_mapping(self) -> dict[str, Any]:
        return {"type": self.type.value, "values": list(self.values)}


@dataclass(frozen=True, slots=True)
class BenchmarkExecution(Mapping[str, Any]):
    """Lower-level benchmark config plus an optional named load dimension."""

    config: dict[str, Any]
    load: ConcurrencySweep | None = None

    @classmethod
    def from_mapping(cls, value: object, *, context: str) -> BenchmarkExecution:
        data = _json_mapping(value, context=context, require_non_empty=True)
        load_value = data.pop("load", None)
        load = (
            None
            if load_value is None
            else ConcurrencySweep.from_mapping(
                load_value,
                context=f"{context}.load",
            )
        )

        forbidden = _find_forbidden_execution_fields(data)
        if forbidden:
            formatted = ", ".join(sorted(forbidden))
            raise BenchmarkSchemaError(
                "benchmark.execution contains fields owned elsewhere in the named "
                f"benchmark contract: {formatted}. Bind target/output fields at run "
                "time and define session_generator per dataset."
            )
        if load is not None:
            _validate_load_traffic_contract(data, context=context)
        return cls(config=data, load=load)

    def to_mapping(self) -> dict[str, Any]:
        result = {
            key: _json_value(value, context=key) for key, value in self.config.items()
        }
        if self.load is not None:
            result["load"] = self.load.to_mapping()
        return result

    def __getitem__(self, key: str) -> Any:
        if key == "load" and self.load is not None:
            return self.load.to_mapping()
        return self.config[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.config
        if self.load is not None:
            yield "load"

    def __len__(self) -> int:
        return len(self.config) + (1 if self.load is not None else 0)


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """Pinned provenance for one dataset selection."""

    kind: str
    uri: str
    revision: str
    split: str
    config: str | None = None
    license: str | None = None
    checksum: str | None = None
    expected_rows: int | None = None

    @classmethod
    def from_mapping(cls, value: object, *, context: str) -> DatasetSource:
        data = _mapping(value, context=context)
        _reject_unknown(
            data,
            {
                "kind",
                "uri",
                "revision",
                "split",
                "config",
                "license",
                "checksum",
                "expected_rows",
            },
            context=context,
        )
        revision = _non_empty_string(
            data.get("revision"), context=f"{context}.revision"
        )
        if revision.lower() in _FLOATING_REVISIONS:
            raise BenchmarkSchemaError(
                f"{context}.revision must pin an immutable revision, not {revision!r}"
            )
        return cls(
            kind=_non_empty_string(data.get("kind"), context=f"{context}.kind"),
            uri=_non_empty_string(data.get("uri"), context=f"{context}.uri"),
            revision=revision,
            split=_non_empty_string(data.get("split"), context=f"{context}.split"),
            config=_optional_string(data.get("config"), context=f"{context}.config"),
            license=_optional_string(data.get("license"), context=f"{context}.license"),
            checksum=_optional_string(
                data.get("checksum"), context=f"{context}.checksum"
            ),
            expected_rows=_optional_positive_int(
                data.get("expected_rows"), context=f"{context}.expected_rows"
            ),
        )


@dataclass(frozen=True, slots=True)
class DatasetCase:
    """A reproducible dataset slice run under a benchmark's common contract."""

    id: str
    name: str
    source: DatasetSource
    session_generator: dict[str, Any]
    client_overrides: dict[str, Any]
    description: str | None = None

    @classmethod
    def from_mapping(cls, value: object, *, index: int) -> DatasetCase:
        context = f"datasets[{index}]"
        data = _mapping(value, context=context)
        _reject_unknown(
            data,
            {
                "id",
                "name",
                "description",
                "source",
                "session_generator",
                "client_overrides",
            },
            context=context,
        )
        session_generator = _json_mapping(
            data.get("session_generator"),
            context=f"{context}.session_generator",
            require_non_empty=True,
        )
        client_overrides = _json_mapping(
            data.get("client_overrides", {}),
            context=f"{context}.client_overrides",
        )
        _validate_client_overrides(
            client_overrides, context=f"{context}.client_overrides"
        )
        return cls(
            id=_stable_id(data.get("id"), context=f"{context}.id"),
            name=_non_empty_string(data.get("name"), context=f"{context}.name"),
            description=_optional_string(
                data.get("description"), context=f"{context}.description"
            ),
            source=DatasetSource.from_mapping(
                data.get("source"), context=f"{context}.source"
            ),
            session_generator=session_generator,
            client_overrides=client_overrides,
        )


@dataclass(frozen=True, slots=True)
class Aggregation:
    """A validated reducer declaration.

    ``source`` names an already-computed child metric for ``scalar`` and
    ``equal_dataset_mean``, or a raw request field for distribution reducers.
    ``ratio_of_sums`` names additive numerator and denominator fields so rates
    such as WER are never averaged with the wrong weighting.
    """

    method: AggregationMethod
    source: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    quantiles: tuple[float, ...] = ()
    scale: float = 1.0

    @classmethod
    def from_mapping(cls, value: object, *, context: str) -> Aggregation:
        data = _mapping(value, context=context)
        _reject_unknown(
            data,
            {"method", "source", "numerator", "denominator", "quantiles", "scale"},
            context=context,
        )
        method = _enum_value(
            AggregationMethod, data.get("method"), context=f"{context}.method"
        )
        source = _optional_string(data.get("source"), context=f"{context}.source")
        numerator = _optional_string(
            data.get("numerator"), context=f"{context}.numerator"
        )
        denominator = _optional_string(
            data.get("denominator"), context=f"{context}.denominator"
        )
        quantiles = _quantiles(data.get("quantiles", ()), context=context)
        scale = _positive_number(data.get("scale", 1.0), context=f"{context}.scale")

        if method in {AggregationMethod.SCALAR, AggregationMethod.EQUAL_DATASET_MEAN}:
            _require(source is not None, f"{context}.source is required for {method}")
            _require(
                numerator is None and denominator is None and not quantiles,
                f"{context}: {method} only accepts source",
            )
            _require(scale == 1.0, f"{context}.scale is only valid for ratio_of_sums")
        elif method in {
            AggregationMethod.DISTRIBUTION,
            AggregationMethod.POOLED_DISTRIBUTION,
        }:
            _require(source is not None, f"{context}.source is required for {method}")
            _require(bool(quantiles), f"{context}.quantiles is required for {method}")
            _require(
                numerator is None and denominator is None,
                f"{context}: {method} does not accept numerator or denominator",
            )
            _require(scale == 1.0, f"{context}.scale is only valid for ratio_of_sums")
        elif method is AggregationMethod.RATIO_OF_SUMS:
            _require(
                numerator is not None and denominator is not None,
                f"{context}.numerator and denominator are required for {method}",
            )
            _require(
                source is None and not quantiles,
                f"{context}: {method} only accepts numerator and denominator",
            )
        elif method is AggregationMethod.NONE:
            _require(
                source is None
                and numerator is None
                and denominator is None
                and not quantiles,
                f"{context}: none does not accept reducer operands",
            )
            _require(scale == 1.0, f"{context}.scale is only valid for ratio_of_sums")

        return cls(
            method=method,
            source=source,
            numerator=numerator,
            denominator=denominator,
            quantiles=quantiles,
            scale=scale,
        )


@dataclass(frozen=True, slots=True)
class Metric:
    id: str
    role: MetricRole
    unit: str
    dataset_aggregation: Aggregation
    benchmark_aggregation: Aggregation
    description: str | None = None
    requires_all_requests_successful: bool = False

    @classmethod
    def from_mapping(cls, value: object, *, index: int) -> Metric:
        context = f"metrics[{index}]"
        data = _mapping(value, context=context)
        _reject_unknown(
            data,
            {
                "id",
                "role",
                "unit",
                "description",
                "requires_all_requests_successful",
                "dataset_aggregation",
                "benchmark_aggregation",
            },
            context=context,
        )
        metric_id = _stable_id(data.get("id"), context=f"{context}.id")
        role = _enum_value(MetricRole, data.get("role"), context=f"{context}.role")
        dataset_aggregation = Aggregation.from_mapping(
            data.get("dataset_aggregation"),
            context=f"{context}.dataset_aggregation",
        )
        benchmark_aggregation = Aggregation.from_mapping(
            data.get("benchmark_aggregation"),
            context=f"{context}.benchmark_aggregation",
        )
        _validate_aggregation_pair(
            metric_id,
            role,
            dataset_aggregation,
            benchmark_aggregation,
            context=context,
        )
        return cls(
            id=metric_id,
            role=role,
            unit=_non_empty_string(data.get("unit"), context=f"{context}.unit"),
            description=_optional_string(
                data.get("description"), context=f"{context}.description"
            ),
            requires_all_requests_successful=_boolean(
                data.get("requires_all_requests_successful", False),
                context=f"{context}.requires_all_requests_successful",
            ),
            dataset_aggregation=dataset_aggregation,
            benchmark_aggregation=benchmark_aggregation,
        )


@dataclass(frozen=True, slots=True)
class Benchmark:
    """Versioned named workload, independent of any provider or deployment."""

    schema_version: int
    id: str
    name: str
    description: str
    modality: Modality
    input_mode: InteractionMode
    output_mode: InteractionMode
    interaction: dict[str, Any]
    client_overrides: dict[str, Any]
    datasets: tuple[DatasetCase, ...]
    execution: BenchmarkExecution
    metrics: tuple[Metric, ...]

    @classmethod
    def from_mapping(cls, value: object) -> Benchmark:
        data = _mapping(value, context="benchmark")
        _reject_unknown(
            data,
            {
                "schema_version",
                "id",
                "name",
                "description",
                "modality",
                "input_mode",
                "output_mode",
                "interaction",
                "client_overrides",
                "datasets",
                "execution",
                "metrics",
            },
            context="benchmark",
        )

        schema_version = data.get("schema_version")
        if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
            raise BenchmarkSchemaError(
                f"benchmark.schema_version must be {SCHEMA_VERSION}, got "
                f"{schema_version!r}"
            )

        interaction = _json_mapping(
            data.get("interaction"),
            context="benchmark.interaction",
            require_non_empty=True,
        )
        if "protocol_profile" in interaction:
            raise BenchmarkSchemaError(
                "benchmark.interaction.protocol_profile is not supported; put the "
                "exact interaction settings directly in interaction"
            )

        client_overrides = _json_mapping(
            data.get("client_overrides", {}),
            context="benchmark.client_overrides",
        )
        _validate_client_overrides(
            client_overrides, context="benchmark.client_overrides"
        )

        execution = BenchmarkExecution.from_mapping(
            data.get("execution"),
            context="benchmark.execution",
        )

        datasets_value = data.get("datasets")
        datasets_sequence = _sequence(datasets_value, context="benchmark.datasets")
        _require(bool(datasets_sequence), "benchmark.datasets must not be empty")
        datasets = tuple(
            DatasetCase.from_mapping(item, index=index)
            for index, item in enumerate(datasets_sequence)
        )
        _require_unique((dataset.id for dataset in datasets), context="dataset")

        metrics_value = data.get("metrics")
        metrics_sequence = _sequence(metrics_value, context="benchmark.metrics")
        _require(bool(metrics_sequence), "benchmark.metrics must not be empty")
        metrics = tuple(
            Metric.from_mapping(item, index=index)
            for index, item in enumerate(metrics_sequence)
        )
        _require_unique((metric.id for metric in metrics), context="metric")
        _require(
            any(metric.role is MetricRole.PRIMARY for metric in metrics),
            "benchmark.metrics must declare at least one primary metric",
        )

        return cls(
            schema_version=schema_version,
            id=_stable_id(data.get("id"), context="benchmark.id"),
            name=_non_empty_string(data.get("name"), context="benchmark.name"),
            description=_non_empty_string(
                data.get("description"), context="benchmark.description"
            ),
            modality=_enum_value(
                Modality, data.get("modality"), context="benchmark.modality"
            ),
            input_mode=_enum_value(
                InteractionMode,
                data.get("input_mode"),
                context="benchmark.input_mode",
            ),
            output_mode=_enum_value(
                InteractionMode,
                data.get("output_mode"),
                context="benchmark.output_mode",
            ),
            interaction=interaction,
            client_overrides=client_overrides,
            datasets=datasets,
            execution=execution,
            metrics=metrics,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical manifest shape used for hashing and provenance."""

        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "modality": self.modality.value,
            "input_mode": self.input_mode.value,
            "output_mode": self.output_mode.value,
            "interaction": _json_value(self.interaction, context="interaction"),
            "client_overrides": _json_value(
                self.client_overrides,
                context="client_overrides",
            ),
            "datasets": [
                {
                    "id": dataset.id,
                    "name": dataset.name,
                    "description": dataset.description,
                    "source": {
                        "kind": dataset.source.kind,
                        "uri": dataset.source.uri,
                        "revision": dataset.source.revision,
                        "split": dataset.source.split,
                        "config": dataset.source.config,
                        "license": dataset.source.license,
                        "checksum": dataset.source.checksum,
                        "expected_rows": dataset.source.expected_rows,
                    },
                    "session_generator": _json_value(
                        dataset.session_generator,
                        context=f"datasets.{dataset.id}.session_generator",
                    ),
                    "client_overrides": _json_value(
                        dataset.client_overrides,
                        context=f"datasets.{dataset.id}.client_overrides",
                    ),
                }
                for dataset in self.datasets
            ],
            "execution": self.execution.to_mapping(),
            "metrics": [
                {
                    "id": metric.id,
                    "role": metric.role.value,
                    "unit": metric.unit,
                    "description": metric.description,
                    "requires_all_requests_successful": (
                        metric.requires_all_requests_successful
                    ),
                    "dataset_aggregation": _aggregation_to_mapping(
                        metric.dataset_aggregation
                    ),
                    "benchmark_aggregation": _aggregation_to_mapping(
                        metric.benchmark_aggregation
                    ),
                }
                for metric in self.metrics
            ],
        }


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkSchemaError(f"{context} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise BenchmarkSchemaError(f"{context} keys must be strings")
    return value


def _sequence(value: object, *, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BenchmarkSchemaError(f"{context} must be a list")
    return value


def _reject_unknown(
    data: Mapping[str, object], allowed: set[str], *, context: str
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise BenchmarkSchemaError(
            f"{context} has unknown field(s): {', '.join(unknown)}"
        )


def _non_empty_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkSchemaError(f"{context} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, context=context)


def _optional_positive_int(value: object, *, context: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise BenchmarkSchemaError(f"{context} must be a positive integer")
    return value


def _positive_sorted_unique_ints(
    value: object,
    *,
    context: str,
) -> tuple[int, ...]:
    values = _sequence(value, context=context)
    if not values:
        raise BenchmarkSchemaError(f"{context} must not be empty")
    parsed: list[int] = []
    for index, item in enumerate(values):
        if type(item) is not int or item <= 0:
            raise BenchmarkSchemaError(f"{context}[{index}] must be a positive integer")
        parsed.append(item)
    if len(set(parsed)) != len(parsed):
        raise BenchmarkSchemaError(f"{context} must contain unique values")
    if parsed != sorted(parsed):
        raise BenchmarkSchemaError(f"{context} must be strictly increasing")
    return tuple(parsed)


def _boolean(value: object, *, context: str) -> bool:
    if type(value) is not bool:
        raise BenchmarkSchemaError(f"{context} must be a boolean")
    return value


def _stable_id(value: object, *, context: str) -> str:
    stable_id = _non_empty_string(value, context=context)
    if not _ID_PATTERN.fullmatch(stable_id):
        raise BenchmarkSchemaError(
            f"{context} must match {_ID_PATTERN.pattern!r}, got {stable_id!r}"
        )
    return stable_id


def _enum_value(enum_type: type[_EnumT], value: object, *, context: str) -> _EnumT:
    if not isinstance(value, str):
        raise BenchmarkSchemaError(f"{context} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise BenchmarkSchemaError(
            f"{context} must be one of: {choices}; got {value!r}"
        ) from exc


def _json_mapping(
    value: object, *, context: str, require_non_empty: bool = False
) -> dict[str, Any]:
    mapping = _mapping(value, context=context)
    if require_non_empty and not mapping:
        raise BenchmarkSchemaError(f"{context} must not be empty")
    result = {
        key: _json_value(item, context=f"{context}.{key}")
        for key, item in mapping.items()
    }
    return result


def _json_value(value: object, *, context: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BenchmarkSchemaError(f"{context} must be a finite number")
        return value
    if isinstance(value, Mapping):
        return _json_mapping(value, context=context)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _json_value(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    raise BenchmarkSchemaError(
        f"{context} must contain only JSON-compatible configuration values"
    )


def _quantiles(value: object, *, context: str) -> tuple[float, ...]:
    values = _sequence(value, context=f"{context}.quantiles")
    quantiles: list[float] = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise BenchmarkSchemaError(f"{context}.quantiles[{index}] must be a number")
        quantile = float(item)
        if not 0.0 < quantile < 1.0:
            raise BenchmarkSchemaError(
                f"{context}.quantiles[{index}] must be between 0 and 1"
            )
        quantiles.append(quantile)
    if len(set(quantiles)) != len(quantiles):
        raise BenchmarkSchemaError(f"{context}.quantiles must be unique")
    if quantiles != sorted(quantiles):
        raise BenchmarkSchemaError(f"{context}.quantiles must be sorted")
    return tuple(quantiles)


def _find_forbidden_execution_fields(
    value: object, path: str = "execution"
) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = key.lower().replace("-", "_") if isinstance(key, str) else key
            child_path = f"{path}.{key}"
            if normalized in _FORBIDDEN_EXECUTION_FIELDS:
                found.add(child_path)
            found.update(_find_forbidden_execution_fields(item, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            found.update(_find_forbidden_execution_fields(item, f"{path}[{index}]"))
    return found


def _validate_load_traffic_contract(
    execution: Mapping[str, Any],
    *,
    context: str,
) -> None:
    traffic = execution.get("traffic_scheduler")
    if not isinstance(traffic, Mapping):
        raise BenchmarkSchemaError(
            f"{context}.load requires an explicit concurrent traffic_scheduler"
        )
    if traffic.get("type") != "concurrent":
        raise BenchmarkSchemaError(
            f"{context}.load requires traffic_scheduler.type=concurrent"
        )
    if "target_concurrent_sessions" in traffic:
        raise BenchmarkSchemaError(
            f"{context}.traffic_scheduler.target_concurrent_sessions conflicts "
            f"with {context}.load.values; declare concurrency only in load.values"
        )


def _aggregation_to_mapping(aggregation: Aggregation) -> dict[str, Any]:
    return {
        "method": aggregation.method.value,
        "source": aggregation.source,
        "numerator": aggregation.numerator,
        "denominator": aggregation.denominator,
        "quantiles": list(aggregation.quantiles),
        "scale": aggregation.scale,
    }


def _validate_client_overrides(overrides: Mapping[str, Any], *, context: str) -> None:
    forbidden = sorted(_TARGET_CLIENT_FIELDS.intersection(overrides))
    if forbidden:
        raise BenchmarkSchemaError(
            f"{context} must describe workload behavior, not bind a target; "
            "move these fields to target_config: " + ", ".join(forbidden)
        )


def _validate_aggregation_pair(
    metric_id: str,
    role: MetricRole,
    dataset: Aggregation,
    benchmark: Aggregation,
    *,
    context: str,
) -> None:
    allowed_benchmark_methods = {
        AggregationMethod.SCALAR: {
            AggregationMethod.EQUAL_DATASET_MEAN,
            AggregationMethod.NONE,
        },
        AggregationMethod.DISTRIBUTION: {
            AggregationMethod.POOLED_DISTRIBUTION,
            AggregationMethod.NONE,
        },
        AggregationMethod.RATIO_OF_SUMS: {
            AggregationMethod.RATIO_OF_SUMS,
            AggregationMethod.NONE,
        },
        AggregationMethod.NONE: {AggregationMethod.NONE},
    }
    if dataset.method not in allowed_benchmark_methods:
        raise BenchmarkSchemaError(
            f"{context}.dataset_aggregation method {dataset.method!r} is not a "
            "dataset-level reducer"
        )
    allowed = allowed_benchmark_methods[dataset.method]
    if benchmark.method not in allowed:
        choices = ", ".join(sorted(method.value for method in allowed))
        raise BenchmarkSchemaError(
            f"metric {metric_id!r}: benchmark aggregation {benchmark.method!r} "
            f"is unsafe after dataset aggregation {dataset.method!r}; use: {choices}"
        )
    if role is MetricRole.PRIMARY and (
        dataset.method is AggregationMethod.NONE
        or benchmark.method is AggregationMethod.NONE
    ):
        raise BenchmarkSchemaError(
            f"primary metric {metric_id!r} must define both dataset and benchmark "
            "aggregation"
        )

    if benchmark.method is AggregationMethod.NONE:
        return
    if dataset.method in {AggregationMethod.SCALAR, AggregationMethod.DISTRIBUTION}:
        if dataset.source != benchmark.source:
            raise BenchmarkSchemaError(
                f"metric {metric_id!r} must use the same source at dataset and "
                "benchmark levels"
            )
    elif dataset.method is AggregationMethod.RATIO_OF_SUMS:
        if (
            dataset.numerator != benchmark.numerator
            or dataset.denominator != benchmark.denominator
            or dataset.scale != benchmark.scale
        ):
            raise BenchmarkSchemaError(
                f"metric {metric_id!r} must use the same ratio operands at dataset "
                "and benchmark levels"
            )


def _positive_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkSchemaError(f"{context} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise BenchmarkSchemaError(f"{context} must be a positive finite number")
    return number


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkSchemaError(message)


def _require_unique(values, *, context: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise BenchmarkSchemaError(
            f"benchmark contains duplicate {context} id(s): "
            + ", ".join(sorted(duplicates))
        )
