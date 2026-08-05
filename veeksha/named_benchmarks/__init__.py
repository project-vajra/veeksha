"""Versioned, provider-independent named benchmark definitions."""

from veeksha.named_benchmarks.catalog import (
    BenchmarkNotFoundError,
    available_benchmarks,
    load_benchmark,
)
from veeksha.named_benchmarks.schema import (
    SCHEMA_VERSION,
    Aggregation,
    AggregationMethod,
    Benchmark,
    BenchmarkSchemaError,
    DatasetCase,
    DatasetSource,
    InteractionMode,
    Metric,
    MetricRole,
    Modality,
)

__all__ = [
    "SCHEMA_VERSION",
    "Aggregation",
    "AggregationMethod",
    "Benchmark",
    "BenchmarkNotFoundError",
    "BenchmarkSchemaError",
    "DatasetCase",
    "DatasetSource",
    "InteractionMode",
    "Metric",
    "MetricRole",
    "Modality",
    "available_benchmarks",
    "load_benchmark",
]
