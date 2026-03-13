from dataclasses import replace

import pytest  # type: ignore[import]

from veeksha.case_studies.workload_shape_search import (
    BenchmarkRunSummary,
    Guardrails,
    ObjectiveWeights,
    _cache_divergence,
    _scrape_vllm_metrics,
    score_paired_candidate,
)


def _summary(
    *,
    workload: str,
    rate: float,
    ttfc_p99_s: float,
    e2e_p95_s: float,
    throughput: float,
    prompt_reuse: float,
    completion_ratio: float = 0.99,
    completed_requests: int = 400,
    error_rate: float = 0.0,
    all_slos_met: bool = True,
) -> BenchmarkRunSummary:
    return BenchmarkRunSummary(
        workload=workload,
        rate=rate,
        run_dir=f"/tmp/{workload}",
        total_requests=completed_requests,
        completed_requests=completed_requests,
        errored_requests=0,
        error_rate=error_rate,
        completion_ratio=completion_ratio,
        all_slos_met=all_slos_met,
        observed_session_dispatch_rate=rate,
        ttfc_p50_s=ttfc_p99_s / 2,
        ttfc_p95_s=ttfc_p99_s * 0.9,
        ttfc_p99_s=ttfc_p99_s,
        e2e_p50_s=e2e_p95_s / 2,
        e2e_p95_s=e2e_p95_s,
        e2e_p99_s=e2e_p95_s * 1.1,
        tpot_mean_s=0.02,
        tpot_based_throughput=throughput,
        tbc_based_throughput=throughput,
        mean_total_prompt_tokens=2000.0,
        mean_delta_prompt_tokens=500.0,
        mean_cacheable_prompt_tokens=1500.0,
        mean_prompt_reuse_ratio=prompt_reuse,
        decode_window_tbc_p99_s=0.03,
        decode_window_duration_s=30.0,
    )


def test_score_prefers_more_divergent_and_more_loaded_healthy_candidate() -> None:
    guardrails = Guardrails(
        min_completed_requests=100,
        min_completion_ratio=0.95,
        max_error_rate=0.02,
        max_ttfc_p99_s=3.0,
        max_e2e_p95_s=20.0,
        require_all_slos_met=True,
    )
    objective = ObjectiveWeights()

    low_load = score_paired_candidate(
        linear=_summary(
            workload="linear",
            rate=2.0,
            ttfc_p99_s=0.30,
            e2e_p95_s=1.0,
            throughput=120.0,
            prompt_reuse=0.45,
        ),
        dag=_summary(
            workload="dag",
            rate=2.0,
            ttfc_p99_s=0.42,
            e2e_p95_s=1.3,
            throughput=100.0,
            prompt_reuse=0.62,
        ),
        guardrails=guardrails,
        objective=objective,
    )
    high_load = score_paired_candidate(
        linear=_summary(
            workload="linear",
            rate=8.0,
            ttfc_p99_s=1.5,
            e2e_p95_s=8.0,
            throughput=115.0,
            prompt_reuse=0.45,
        ),
        dag=_summary(
            workload="dag",
            rate=8.0,
            ttfc_p99_s=2.4,
            e2e_p95_s=12.0,
            throughput=78.0,
            prompt_reuse=0.68,
        ),
        guardrails=guardrails,
        objective=objective,
    )

    assert low_load.healthy is True
    assert high_load.healthy is True
    assert high_load.divergence_score > low_load.divergence_score
    assert high_load.load_factor > low_load.load_factor
    assert high_load.overall_score > low_load.overall_score


def test_score_marks_candidate_unhealthy_when_guardrails_fail() -> None:
    result = score_paired_candidate(
        linear=_summary(
            workload="linear",
            rate=16.0,
            ttfc_p99_s=1.8,
            e2e_p95_s=9.0,
            throughput=110.0,
            prompt_reuse=0.44,
        ),
        dag=_summary(
            workload="dag",
            rate=16.0,
            ttfc_p99_s=4.2,
            e2e_p95_s=24.0,
            throughput=60.0,
            prompt_reuse=0.70,
            completion_ratio=0.80,
            error_rate=0.08,
            all_slos_met=False,
        ),
        guardrails=Guardrails(),
        objective=ObjectiveWeights(),
    )

    assert result.healthy is False
    assert result.status == "guardrail_failed"
    assert result.overall_score == 0.0
    assert any("dag: completion_ratio" in note for note in result.notes)
    assert any("dag: error_rate" in note for note in result.notes)
    assert any("dag: ttfc_p99" in note for note in result.notes)
    assert any("dag: e2e_p95" in note for note in result.notes)
    assert any("dag: benchmark SLOs not met" == note for note in result.notes)


def test_cache_divergence_uses_vllm_cache_metrics_when_available() -> None:
    linear = replace(
        _summary(
            workload="linear",
            rate=6.0,
            ttfc_p99_s=1.0,
            e2e_p95_s=5.0,
            throughput=100.0,
            prompt_reuse=0.90,
        ),
        vllm_prefix_cache_hit_rate=0.75,
        vllm_prompt_cache_token_ratio=0.80,
        vllm_kv_cache_usage_perc=0.35,
    )
    dag = replace(
        _summary(
            workload="dag",
            rate=6.0,
            ttfc_p99_s=1.4,
            e2e_p95_s=7.5,
            throughput=82.0,
            prompt_reuse=0.91,
        ),
        vllm_prefix_cache_hit_rate=0.30,
        vllm_prompt_cache_token_ratio=0.45,
        vllm_kv_cache_usage_perc=0.78,
    )

    divergence = _cache_divergence(linear=linear, dag=dag)

    assert divergence == pytest.approx((0.45 + 0.35 + 0.43) / 3.0)


def test_scrape_vllm_metrics_persists_summary(tmp_path) -> None:
    metrics_text = """
# HELP vllm metrics
vllm:kv_cache_usage_perc 0.625
vllm:prefix_cache_hits 9
vllm:prefix_cache_queries 12
vllm:prompt_tokens_cached 360
vllm:prompt_tokens_recomputed 90
vllm:num_preemptions 1
""".strip()

    class _Response:
        status_code = 200
        text = metrics_text

        def raise_for_status(self) -> None:
            return None

    from unittest.mock import patch

    with patch(
        "veeksha.case_studies.workload_shape_search.requests.get",
        return_value=_Response(),
    ) as mock_get:
        summary = _scrape_vllm_metrics(
            run_dir=str(tmp_path),
            metrics_url="http://localhost:18000/metrics",
            scrape_timeout_s=3.0,
        )

    mock_get.assert_called_once_with("http://localhost:18000/metrics", timeout=3.0)
    assert summary["kv_cache_usage_perc"] == pytest.approx(0.625)
    assert summary["prefix_cache_hit_rate"] == pytest.approx(0.75)
    assert summary["prompt_cache_token_ratio"] == pytest.approx(0.80)
    assert summary["num_preemptions"] == pytest.approx(1.0)
    assert (tmp_path / "metrics" / "vllm_metrics.prom").exists()
    assert (tmp_path / "metrics" / "vllm_metrics_summary.json").exists()
