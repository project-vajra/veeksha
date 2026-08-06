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
    BenchmarkExecution,
    BenchmarkSchemaError,
    ConcurrencyLoadPoint,
    ConcurrencySweep,
    DatasetCase,
    DatasetSource,
    InteractionMode,
    LoadType,
    Metric,
    MetricRole,
    Modality,
)

__all__ = [
    "SCHEMA_VERSION",
    "Aggregation",
    "AggregationMethod",
    "Benchmark",
    "BenchmarkExecution",
    "BenchmarkNotFoundError",
    "BenchmarkSchemaError",
    "ConcurrencyLoadPoint",
    "ConcurrencySweep",
    "DatasetCase",
    "DatasetSource",
    "InteractionMode",
    "LoadType",
    "Metric",
    "MetricRole",
    "Modality",
    "available_benchmarks",
    "load_benchmark",
]
