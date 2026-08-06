from __future__ import annotations

import json
from pathlib import Path

import pytest

from veeksha.named_benchmarks import Benchmark
from veeksha.named_benchmarks.aggregation import (
    CompletedDatasetRun,
    build_named_benchmark_results,
    build_named_benchmark_sweep_results,
)
from veeksha.named_benchmarks.schema import ConcurrencyLoadPoint


def _benchmark() -> Benchmark:
    return Benchmark.from_mapping(
        {
            "schema_version": 1,
            "id": "asr.indic.results-fixture.v1",
            "name": "Results fixture",
            "description": "Exercises exact dataset and cross-dataset reducers.",
            "modality": "asr",
            "input_mode": "streaming",
            "output_mode": "streaming",
            "interaction": {"input_pacing": "realtime"},
            "datasets": [
                {
                    "id": dataset_id,
                    "name": dataset_id,
                    "source": {
                        "kind": "fixture",
                        "uri": f"fixture://{dataset_id}",
                        "revision": "0123456789abcdef",
                        "split": "test",
                    },
                    "session_generator": {
                        "type": "trace",
                        "trace_file": f"{dataset_id}.jsonl",
                    },
                }
                for dataset_id in ("dataset_a", "dataset_b")
            ],
            "execution": {"runtime": {"max_sessions": -1}},
            "metrics": [
                {
                    "id": "first_visible_transcript_ms",
                    "role": "primary",
                    "unit": "ms",
                    "dataset_aggregation": {
                        "method": "distribution",
                        "source": "time_to_first_visible_text",
                        "quantiles": [0.5, 0.9],
                    },
                    "benchmark_aggregation": {
                        "method": "pooled_distribution",
                        "source": "time_to_first_visible_text",
                        "quantiles": [0.5, 0.9],
                    },
                },
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
                },
            ],
        }
    )


def _child(
    tmp_path: Path,
    *,
    dataset_id: str,
    latencies: list[float],
    errors: int,
    reference_words: int,
    load_point_id: str | None = None,
    concurrency: int | None = None,
) -> CompletedDatasetRun:
    run_dir = tmp_path / (load_point_id or "legacy") / dataset_id
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    with (metrics_dir / "request_level_metrics.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for latency in latencies:
            file.write(json.dumps({"time_to_first_visible_text": latency}) + "\n")
    return CompletedDatasetRun(
        target_id="provider-model",
        dataset_id=dataset_id,
        run_dir=run_dir,
        metrics={
            "Number of Requests": len(latencies),
            "Number of Completed Requests": len(latencies),
            "Number of Errored Requests": 0,
            "Number of Cancelled Requests": 0,
            "asr_final_errors": errors,
            "asr_final_reference_words": reference_words,
        },
        load_point_id=load_point_id,
        load=(
            None
            if concurrency is None
            else {
                "type": "concurrency",
                "target_concurrent_sessions": concurrency,
            }
        ),
    )


def test_dataset_metrics_are_preserved_and_cross_dataset_metrics_are_exact(
    tmp_path: Path,
) -> None:
    # Dataset percentiles are 0 ms and 100 ms. Their arithmetic mean is 50 ms,
    # but 8/10 pooled request observations are 100 ms, so pooled P90 is 100 ms.
    dataset_a = _child(
        tmp_path,
        dataset_id="dataset_a",
        latencies=[0, 0],
        errors=1,
        reference_words=1,
    )
    dataset_b = _child(
        tmp_path,
        dataset_id="dataset_b",
        latencies=[100] * 8,
        errors=9,
        reference_words=99,
    )

    result = build_named_benchmark_results(
        _benchmark(),
        [dataset_b, dataset_a],
    )

    target = result["targets"][0]
    assert target["sample_count"] == 10
    assert target["dataset_count"] == 2
    assert target["missing_metric_diagnostics"] == []

    datasets = {row["dataset_id"]: row for row in target["datasets"]}
    assert (
        datasets["dataset_a"]["resolved_metrics"]["first_visible_transcript_ms"][
            "quantiles"
        ]["p90"]
        == 0
    )
    assert (
        datasets["dataset_b"]["resolved_metrics"]["first_visible_transcript_ms"][
            "quantiles"
        ]["p90"]
        == 100
    )
    assert datasets["dataset_a"]["resolved_metrics"]["final_corpus_wer"]["value"] == 100
    assert datasets["dataset_b"]["resolved_metrics"]["final_corpus_wer"][
        "value"
    ] == pytest.approx(9.090909)

    pooled_latency = target["benchmark_metrics"]["first_visible_transcript_ms"]
    assert pooled_latency["observation_count"] == 10
    assert pooled_latency["quantiles"]["p90"] == 100
    assert pooled_latency["quantiles"]["p90"] != 50

    corpus_wer = target["benchmark_metrics"]["final_corpus_wer"]
    assert corpus_wer["numerator_sum"] == 10
    assert corpus_wer["denominator_sum"] == 100
    assert corpus_wer["scale"] == 100
    assert corpus_wer["value"] == 10


def test_results_report_missing_dataset_runs_without_hiding_completed_data(
    tmp_path: Path,
) -> None:
    result = build_named_benchmark_results(
        _benchmark(),
        [
            _child(
                tmp_path,
                dataset_id="dataset_a",
                latencies=[12],
                errors=0,
                reference_words=3,
            )
        ],
    )

    target = result["targets"][0]
    assert [dataset["dataset_id"] for dataset in target["datasets"]] == ["dataset_a"]
    assert any(
        diagnostic.get("reason") == "missing_dataset_run"
        and diagnostic.get("dataset_id") == "dataset_b"
        for diagnostic in target["missing_metric_diagnostics"]
    )
    for metric in target["benchmark_metrics"].values():
        assert metric["dataset_count"] == 1
        assert metric["expected_dataset_count"] == 2
        assert metric["complete"] is False


def test_duplicate_target_dataset_child_is_rejected(tmp_path: Path) -> None:
    child = _child(
        tmp_path,
        dataset_id="dataset_a",
        latencies=[1],
        errors=0,
        reference_words=1,
    )

    with pytest.raises(ValueError, match="duplicate found"):
        build_named_benchmark_results(_benchmark(), [child, child])


def test_correctness_metrics_are_withheld_when_any_request_failed(
    tmp_path: Path,
) -> None:
    child = _child(
        tmp_path,
        dataset_id="dataset_a",
        latencies=[12, 14],
        errors=0,
        reference_words=10,
    )
    metrics = dict(child.metrics)
    metrics["Number of Errored Requests"] = 1
    child = CompletedDatasetRun(
        target_id=child.target_id,
        dataset_id=child.dataset_id,
        run_dir=child.run_dir,
        metrics=metrics,
    )

    result = build_named_benchmark_results(_benchmark(), [child])
    target = result["targets"][0]

    assert "final_corpus_wer" not in target["datasets"][0]["resolved_metrics"]
    assert "final_corpus_wer" not in target["benchmark_metrics"]
    assert any(
        diagnostic["reason"] == "metric_invalid_due_to_unsuccessful_requests"
        for diagnostic in target["missing_metric_diagnostics"]
    )


@pytest.mark.parametrize(
    ("completed", "errored", "cancelled"),
    [(1, 0, 1), (1, 0, 0)],
)
def test_correctness_metrics_are_withheld_for_cancelled_or_incomplete_requests(
    tmp_path: Path,
    completed: int,
    errored: int,
    cancelled: int,
) -> None:
    child = _child(
        tmp_path,
        dataset_id="dataset_a",
        latencies=[12, 14],
        errors=0,
        reference_words=10,
    )
    metrics = {
        **child.metrics,
        "Number of Completed Requests": completed,
        "Number of Errored Requests": errored,
        "Number of Cancelled Requests": cancelled,
    }
    child = CompletedDatasetRun(
        target_id=child.target_id,
        dataset_id=child.dataset_id,
        run_dir=child.run_dir,
        metrics=metrics,
    )

    result = build_named_benchmark_results(_benchmark(), [child])
    target = result["targets"][0]

    assert "final_corpus_wer" not in target["datasets"][0]["resolved_metrics"]
    assert "final_corpus_wer" not in target["benchmark_metrics"]
    assert any(
        diagnostic["reason"] == "metric_invalid_due_to_unsuccessful_requests"
        for diagnostic in target["missing_metric_diagnostics"]
    )


@pytest.mark.parametrize(
    ("counter", "value"),
    [("Number of Completed Requests", None), ("Number of Requests", 2.5)],
)
def test_correctness_metrics_fail_closed_for_missing_or_invalid_lifecycle_counts(
    tmp_path: Path,
    counter: str,
    value: float | None,
) -> None:
    child = _child(
        tmp_path,
        dataset_id="dataset_a",
        latencies=[12, 14],
        errors=0,
        reference_words=10,
    )
    metrics = dict(child.metrics)
    if value is None:
        metrics.pop(counter)
    else:
        metrics[counter] = value
    child = CompletedDatasetRun(
        target_id=child.target_id,
        dataset_id=child.dataset_id,
        run_dir=child.run_dir,
        metrics=metrics,
    )

    result = build_named_benchmark_results(_benchmark(), [child])
    target = result["targets"][0]

    assert "final_corpus_wer" not in target["benchmark_metrics"]
    assert any(
        diagnostic["reason"] == "request_lifecycle_counts_unavailable"
        for diagnostic in target["missing_metric_diagnostics"]
    )


def test_expected_target_is_retained_when_all_child_runs_fail() -> None:
    result = build_named_benchmark_results(
        _benchmark(),
        [],
        expected_target_ids=["failed-provider-model"],
    )

    assert [target["target_id"] for target in result["targets"]] == [
        "failed-provider-model"
    ]
    target = result["targets"][0]
    assert target["dataset_count"] == 0
    assert target["expected_dataset_count"] == 2
    assert target["benchmark_metrics"] == {}
    assert {
        diagnostic.get("dataset_id")
        for diagnostic in target["missing_metric_diagnostics"]
        if diagnostic["reason"] == "missing_dataset_run"
    } == {"dataset_a", "dataset_b"}


def test_sweep_aggregates_each_concurrency_without_cross_load_pooling(
    tmp_path: Path,
) -> None:
    load_1 = ConcurrencyLoadPoint(
        id="concurrency-0001",
        target_concurrent_sessions=1,
    )
    load_8 = ConcurrencyLoadPoint(
        id="concurrency-0008",
        target_concurrent_sessions=8,
    )
    children = [
        _child(
            tmp_path,
            dataset_id="dataset_a",
            latencies=[10, 10],
            errors=0,
            reference_words=10,
            load_point_id=load_1.id,
            concurrency=1,
        ),
        _child(
            tmp_path,
            dataset_id="dataset_b",
            latencies=[20, 20],
            errors=1,
            reference_words=90,
            load_point_id=load_1.id,
            concurrency=1,
        ),
        _child(
            tmp_path,
            dataset_id="dataset_a",
            latencies=[100, 100],
            errors=5,
            reference_words=10,
            load_point_id=load_8.id,
            concurrency=8,
        ),
        _child(
            tmp_path,
            dataset_id="dataset_b",
            latencies=[200, 200],
            errors=45,
            reference_words=90,
            load_point_id=load_8.id,
            concurrency=8,
        ),
    ]

    result = build_named_benchmark_sweep_results(
        _benchmark(),
        children,
        load_points=[load_1, load_8],
        expected_target_ids=["provider-model"],
    )

    assert result["schema_version"] == 2
    target = result["targets"][0]
    assert "benchmark_metrics" not in target
    points = {point["load_point_id"]: point for point in target["load_points"]}
    assert list(points) == ["concurrency-0001", "concurrency-0008"]
    assert (
        points["concurrency-0001"]["benchmark_metrics"]["first_visible_transcript_ms"][
            "quantiles"
        ]["p90"]
        == 20
    )
    assert (
        points["concurrency-0008"]["benchmark_metrics"]["first_visible_transcript_ms"][
            "quantiles"
        ]["p90"]
        == 200
    )
    assert (
        points["concurrency-0001"]["benchmark_metrics"]["final_corpus_wer"]["value"]
        == 1
    )
    assert (
        points["concurrency-0008"]["benchmark_metrics"]["final_corpus_wer"]["value"]
        == 50
    )


def test_sweep_retains_empty_expected_load_point_with_its_own_diagnostics(
    tmp_path: Path,
) -> None:
    load_1 = ConcurrencyLoadPoint(
        id="concurrency-0001",
        target_concurrent_sessions=1,
    )
    load_8 = ConcurrencyLoadPoint(
        id="concurrency-0008",
        target_concurrent_sessions=8,
    )
    child = _child(
        tmp_path,
        dataset_id="dataset_a",
        latencies=[10],
        errors=0,
        reference_words=10,
        load_point_id=load_1.id,
        concurrency=1,
    )

    result = build_named_benchmark_sweep_results(
        _benchmark(),
        [child],
        load_points=[load_1, load_8],
        expected_target_ids=["provider-model"],
    )

    points = {
        point["load_point_id"]: point for point in result["targets"][0]["load_points"]
    }
    assert points["concurrency-0001"]["dataset_count"] == 1
    assert points["concurrency-0008"]["dataset_count"] == 0
    assert {
        diagnostic.get("dataset_id")
        for diagnostic in points["concurrency-0008"]["missing_metric_diagnostics"]
        if diagnostic["reason"] == "missing_dataset_run"
    } == {"dataset_a", "dataset_b"}
    assert all(
        diagnostic["load_point_id"] == "concurrency-0008"
        for diagnostic in points["concurrency-0008"]["missing_metric_diagnostics"]
    )


def test_sweep_rejects_child_with_mismatched_load_metadata(tmp_path: Path) -> None:
    load_point = ConcurrencyLoadPoint(
        id="concurrency-0008",
        target_concurrent_sessions=8,
    )
    child = _child(
        tmp_path,
        dataset_id="dataset_a",
        latencies=[10],
        errors=0,
        reference_words=10,
        load_point_id=load_point.id,
        concurrency=4,
    )

    with pytest.raises(ValueError, match="load metadata does not match"):
        build_named_benchmark_sweep_results(
            _benchmark(),
            [child],
            load_points=[load_point],
            expected_target_ids=["provider-model"],
        )
