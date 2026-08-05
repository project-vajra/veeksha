from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from veeksha.named_benchmarks import (
    Benchmark,
    BenchmarkSchemaError,
    available_benchmarks,
    load_benchmark,
)


def _valid_manifest() -> dict:
    return {
        "schema_version": 1,
        "id": "asr.indic.fixture.v1",
        "name": "Indic ASR fixture",
        "description": "A small schema fixture with two independent datasets.",
        "modality": "asr",
        "input_mode": "streaming",
        "output_mode": "streaming",
        "interaction": {
            "audio_encoding": "pcm_s16le",
            "sample_rate_hz": 16000,
            "input_frame_ms": 20,
        },
        "client_overrides": {
            "sample_rate": 16000,
            "ws_realtime_pacing": True,
        },
        "datasets": [
            {
                "id": "dataset_a",
                "name": "Dataset A",
                "source": {
                    "kind": "huggingface",
                    "uri": "example/dataset",
                    "revision": "0123456789abcdef",
                    "config": "a",
                    "split": "test",
                },
                "session_generator": {
                    "type": "trace",
                    "trace_file": "${DATASET_ROOT}/a/manifest.jsonl",
                    "flavor": {"type": "audio", "audio_dir": ""},
                },
            },
            {
                "id": "dataset_b",
                "name": "Dataset B",
                "source": {
                    "kind": "huggingface",
                    "uri": "example/dataset",
                    "revision": "0123456789abcdef",
                    "config": "b",
                    "split": "test",
                },
                "session_generator": {
                    "type": "trace",
                    "trace_file": "${DATASET_ROOT}/b/manifest.jsonl",
                    "flavor": {"type": "audio", "audio_dir": ""},
                },
            },
        ],
        "execution": {
            "traffic_scheduler": {
                "type": "concurrent",
                "target_concurrent_sessions": 1,
            },
            "runtime": {"max_sessions": -1},
        },
        "metrics": [
            {
                "id": "final_corpus_wer",
                "role": "primary",
                "unit": "percent",
                "requires_all_requests_successful": True,
                "dataset_aggregation": {
                    "method": "ratio_of_sums",
                    "numerator": "asr_final_errors",
                    "denominator": "asr_final_reference_words",
                    "scale": 100,
                },
                "benchmark_aggregation": {
                    "method": "ratio_of_sums",
                    "numerator": "asr_final_errors",
                    "denominator": "asr_final_reference_words",
                    "scale": 100,
                },
            }
        ],
    }


def test_valid_manifest_preserves_dataset_and_metric_contract() -> None:
    benchmark = Benchmark.from_mapping(_valid_manifest())

    assert benchmark.id == "asr.indic.fixture.v1"
    assert [dataset.id for dataset in benchmark.datasets] == [
        "dataset_a",
        "dataset_b",
    ]
    assert benchmark.metrics[0].dataset_aggregation.scale == 100.0
    assert benchmark.metrics[0].benchmark_aggregation.scale == 100.0
    assert benchmark.metrics[0].requires_all_requests_successful is True


def test_requires_all_requests_successful_must_be_boolean() -> None:
    manifest = _valid_manifest()
    manifest["metrics"][0]["requires_all_requests_successful"] = "true"

    with pytest.raises(BenchmarkSchemaError, match="must be a boolean"):
        Benchmark.from_mapping(manifest)


@pytest.mark.parametrize("revision", ["main", "master", "latest", "HEAD"])
def test_dataset_source_requires_immutable_revision(revision: str) -> None:
    manifest = _valid_manifest()
    manifest["datasets"][0]["source"]["revision"] = revision

    with pytest.raises(BenchmarkSchemaError, match="immutable revision"):
        Benchmark.from_mapping(manifest)


@pytest.mark.parametrize(
    "field",
    [
        "type",
        "provider",
        "model",
        "api_base",
        "api_key_env",
        "language",
        "language_mode",
        "supported_languages",
    ],
)
def test_manifest_cannot_bind_a_target(field: str) -> None:
    manifest = _valid_manifest()
    manifest["client_overrides"][field] = "provider-specific-value"

    with pytest.raises(BenchmarkSchemaError, match="must describe workload behavior"):
        Benchmark.from_mapping(manifest)


def test_execution_rejects_fields_owned_by_dataset_or_run() -> None:
    manifest = _valid_manifest()
    manifest["execution"]["nested"] = {"session_generator": {"type": "synthetic"}}

    with pytest.raises(
        BenchmarkSchemaError, match="execution.nested.session_generator"
    ):
        Benchmark.from_mapping(manifest)


def test_ratio_scale_must_match_at_dataset_and_benchmark_levels() -> None:
    manifest = _valid_manifest()
    manifest["metrics"][0]["benchmark_aggregation"]["scale"] = 1

    with pytest.raises(BenchmarkSchemaError, match="same ratio operands"):
        Benchmark.from_mapping(manifest)


def test_primary_metric_requires_dataset_and_benchmark_aggregation() -> None:
    manifest = _valid_manifest()
    manifest["metrics"][0]["benchmark_aggregation"] = {"method": "none"}

    with pytest.raises(BenchmarkSchemaError, match="primary metric"):
        Benchmark.from_mapping(manifest)


def test_manifest_rejects_duplicate_dataset_ids() -> None:
    manifest = _valid_manifest()
    duplicate = deepcopy(manifest["datasets"][0])
    manifest["datasets"].append(duplicate)

    with pytest.raises(BenchmarkSchemaError, match="duplicate dataset id"):
        Benchmark.from_mapping(manifest)


def test_packaged_named_benchmarks_are_schema_valid() -> None:
    benchmark_ids = available_benchmarks()

    assert {
        "asr.indic.multidomain16.v1",
        "tts.indic.robustness11.static-stream.v1",
    }.issubset(benchmark_ids)
    for benchmark_id in benchmark_ids:
        assert load_benchmark(benchmark_id).id == benchmark_id


def test_yaml_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yml"
    path.write_text(
        "schema_version: 1\nschema_version: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkSchemaError, match="duplicate YAML key"):
        load_benchmark(path)
