import json

import pytest

from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.evaluator.performance.text import TextPerformanceEvaluator
from veeksha.types import ChannelModality


@pytest.mark.unit
def test_text_prefill_stats_groups_by_target_prompt_tokens(tmp_path) -> None:
    cfg = PerformanceEvaluatorConfig(stream_metrics=False)
    evaluator = TextPerformanceEvaluator(
        config=cfg,
        channel_config=cfg.text_channel,
        benchmark_start_time=0.0,
    )

    # Two prompt lengths (8 and 16), two samples each.
    prompts = [(0, 8, 0.10), (1, 8, 0.20), (2, 16, 0.30), (3, 16, 0.40)]
    for request_id, target_prompt_tokens, ttfc in prompts:
        evaluator.register_request(
            request_id=request_id,
            session_id=0,
            dispatched_at=0.0,
            content=TextChannelRequestContent(
                input_text="hello",
                target_prompt_tokens=target_prompt_tokens,
            ),
            requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=2)),
        )

        result = RequestResult(
            request_id=request_id,
            session_id=0,
            session_total_requests=1,
            channels={
                ChannelModality.TEXT: ChannelResponse(
                    modality=ChannelModality.TEXT,
                    content="xy",
                    metrics={
                        "is_stream": True,
                        "inter_chunk_times": [ttfc, 0.01],
                        "num_delta_prompt_tokens": target_prompt_tokens,
                        "num_total_prompt_tokens": target_prompt_tokens,
                        "num_output_tokens": 2,
                    },
                )
            },
            success=True,
            client_completed_at=ttfc + 0.01,
            client_picked_up_at=0.0,
            scheduler_ready_at=0.0,
            scheduler_dispatched_at=0.0,
            result_processed_at=ttfc + 0.01,
        )

        evaluator.record_request_completed(
            request_id=request_id,
            session_id=0,
            completed_at=ttfc + 0.01,
            response=result,
        )

    evaluator._save_prefill_stats(str(tmp_path))
    out_path = tmp_path / "prefill_stats.json"
    assert out_path.exists()

    data = json.loads(out_path.read_text())
    assert data["metric"] == "ttfc"
    assert data["group_by"] == "target_num_delta_prompt_tokens"

    assert data["groups"]["8"]["count"] == 2
    assert data["groups"]["8"]["mean"] == pytest.approx(0.15)
    assert data["groups"]["16"]["count"] == 2
    assert data["groups"]["16"]["mean"] == pytest.approx(0.35)



