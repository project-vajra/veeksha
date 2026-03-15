import json
from pathlib import Path

from veeksha.case_studies.workload_deployment_claim_search import (
    Guardrails,
    RateModel,
    RateModelWorkload,
    VllmMetricsConfig,
    WorkloadDeploymentClaimConfig,
    WorkloadSearchResult,
    _evaluate_guardrails,
    _load_rate_model,
    _persist_results,
    _rate_summary_for_workload,
)
from veeksha.case_studies.workload_shape_search import BenchmarkRunSummary
from veeksha.case_studies.workload_shape_search import summarize_run


def make_summary(**overrides) -> BenchmarkRunSummary:
    payload = {
        "workload": "linear",
        "rate": 0.2,
        "run_dir": "/tmp/run",
        "total_requests": 200,
        "completed_requests": 200,
        "errored_requests": 0,
        "error_rate": 0.0,
        "completion_ratio": 1.0,
        "all_slos_met": True,
        "observed_session_dispatch_rate": 0.2,
        "ttfc_p50_s": 0.1,
        "ttfc_p95_s": 0.2,
        "ttfc_p99_s": 0.3,
        "e2e_p50_s": 1.0,
        "e2e_p95_s": 2.0,
        "e2e_p99_s": 3.0,
        "tpot_mean_s": 0.01,
        "tpot_based_throughput": 100.0,
        "tbc_based_throughput": 50.0,
        "mean_total_prompt_tokens": 2000.0,
        "mean_delta_prompt_tokens": 500.0,
        "mean_cacheable_prompt_tokens": 1500.0,
        "mean_prompt_reuse_ratio": 0.75,
        "decode_window_tbc_p95_s": 0.08,
        "decode_window_tbc_p99_s": 0.1,
        "decode_window_duration_s": 5.0,
        "vllm_metrics_scraped": True,
        "vllm_metrics_url": "http://localhost/metrics",
        "vllm_metrics_scraped_at_utc": "2026-03-15T00:00:00+00:00",
        "vllm_metrics_scrape_error": None,
        "vllm_kv_cache_usage_perc": 0.0,
        "vllm_prompt_tokens_cached": 1000.0,
        "vllm_prompt_tokens_recomputed": 0.0,
        "vllm_prompt_cache_token_ratio": 1.0,
        "vllm_prefix_cache_hits": 1000.0,
        "vllm_prefix_cache_queries": 1500.0,
        "vllm_prefix_cache_hit_rate": 2.0 / 3.0,
        "vllm_num_preemptions": 0.0,
    }
    payload.update(overrides)
    return BenchmarkRunSummary(**payload)


def test_load_rate_model_and_conversion(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "rate_model": {
                    "rate_basis": "request_rate",
                    "fresh_input_tokens_per_request": 500,
                    "output_tokens_per_request": 300,
                    "workloads": {
                        "linear": {"requests_per_session": 5},
                        "dag": {"requests_per_session": 15},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    rate_model = _load_rate_model(str(metadata_path))
    linear_summary = _rate_summary_for_workload(
        rate_model,
        workload="linear",
        normalized_request_rate=1.0,
    )
    dag_summary = _rate_summary_for_workload(
        rate_model,
        workload="dag",
        normalized_request_rate=1.0,
    )

    assert rate_model.rate_basis == "request_rate"
    assert linear_summary["derived_session_rate"] == 0.2
    assert dag_summary["derived_session_rate"] == 1.0 / 15.0
    assert linear_summary["fresh_input_tokens_per_s"] == 500.0
    assert dag_summary["requested_output_tokens_per_s"] == 300.0


def test_evaluate_guardrails_checks_decode_window_tbc() -> None:
    healthy, notes = _evaluate_guardrails(
        make_summary(decode_window_tbc_p95_s=0.06),
        Guardrails(max_tbc_p95_s=0.05),
    )

    assert not healthy
    assert any("tbc_p95" in note for note in notes)


def test_persist_results_records_normalized_rate_fields(tmp_path: Path) -> None:
    config = WorkloadDeploymentClaimConfig(
        output_dir=str(tmp_path),
        linear_benchmark_config="linear.yml",
        dag_benchmark_config="dag.yml",
        trace_metadata_path="metadata.json",
        rate_basis="request_rate",
        start_value=1.0,
        max_value=4.0,
        expansion_factor=2.0,
        max_iterations=10,
        precision=2,
        guardrails=Guardrails(),
        vllm_metrics=VllmMetricsConfig(),
        gpu_hour_price_usd=None,
    )
    rate_model = RateModel(
        rate_basis="request_rate",
        fresh_input_tokens_per_request=500,
        output_tokens_per_request=300,
        workloads={
            "linear": RateModelWorkload(
                requests_per_session=5,
                fresh_input_tokens_per_session=2500,
                output_tokens_per_session=1500,
            ),
            "dag": RateModelWorkload(
                requests_per_session=15,
                fresh_input_tokens_per_session=7500,
                output_tokens_per_session=4500,
            ),
        },
    )
    result = WorkloadSearchResult(
        workload="linear",
        normalized_request_rate=1.0,
        derived_session_rate=0.2,
        fresh_input_tokens_per_s=500.0,
        requested_output_tokens_per_s=300.0,
        run=make_summary(),
        healthy=True,
        status="healthy",
        notes=[],
    )

    _persist_results(
        config=config,
        rate_model=rate_model,
        workload="linear",
        mode="search",
        output_dir=str(tmp_path),
        results=[result],
        requested_normalized_request_rate=None,
    )

    payload = json.loads(
        (tmp_path / "workload_deployment_claim_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["best_normalized_request_rate"] == 1.0
    assert payload["results"][0]["normalized_request_rate"] == 1.0
    assert payload["results"][0]["derived_session_rate"] == 0.2
    assert payload["results"][0]["fresh_input_tokens_per_s"] == 500.0
    assert payload["results"][0]["requested_output_tokens_per_s"] == 300.0


def test_summarize_run_falls_back_to_slo_tbc_p95(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "run" / "metrics"
    metrics_dir.mkdir(parents=True)

    (metrics_dir / "summary_stats.json").write_text(
        json.dumps(
            {
                "Number of Requests": 10,
                "Number of Completed Requests": 10,
                "Number of Errored Requests": 0,
                "Error Rate": 0.0,
                "Observed Session Dispatch Rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "throughput_metrics.json").write_text(
        json.dumps(
            {
                "tpot_based_throughput": 10.0,
                "tbc_based_throughput": 5.0,
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "slo_results.json").write_text(
        json.dumps(
            {
                "all_slos_met": True,
                "results": [
                    {
                        "slo_metric_key": "tbc_p95",
                        "observed_value": 0.01498,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "request_level_metrics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "ttfc": 0.1,
                    "end_to_end_latency": 1.0,
                    "tpot": 0.01,
                    "num_total_prompt_tokens": 1000,
                    "num_delta_prompt_tokens": 500,
                }
            )
            for _ in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "decode_window_metrics.json").write_text(
        json.dumps(
            {
                "windows": {"total_duration_s": 1.0},
                "tbc_in_window_stats": {
                    "count": 10,
                    "mean": 0.012,
                    "median": 0.012,
                    "p90": 0.014,
                    "p99": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_run(workload="linear", rate=0.5, run_dir=str(tmp_path / "run"))

    assert summary.decode_window_tbc_p95_s == 0.01498
    assert summary.decode_window_tbc_p99_s == 0.02
