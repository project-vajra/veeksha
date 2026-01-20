import json

import pytest

from veeksha.config.evaluator import DecodeWindowConfig, PerformanceEvaluatorConfig
from veeksha.config.evaluator import TextChannelPerformanceConfig
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.evaluator.performance.text import TextPerformanceEvaluator
from veeksha.types import ChannelModality


@pytest.mark.unit
def test_text_decode_window_writes_expected_stats(tmp_path) -> None:
    text_cfg = TextChannelPerformanceConfig(
        decode_window_enabled=True,
        decode_window_config=DecodeWindowConfig(
            min_active_requests=2,
            selection_strategy="longest",
            anchor_to_client_pickup=True,
            require_streaming=True,
        ),
    )
    cfg = PerformanceEvaluatorConfig(stream_metrics=False, text_channel=text_cfg)
    evaluator = TextPerformanceEvaluator(
        config=cfg,
        channel_config=cfg.text_channel,
        benchmark_start_time=0.0,
    )

    for request_id in (0, 1):
        evaluator.register_request(
            request_id=request_id,
            session_id=0,
            dispatched_at=0.0,
            content=TextChannelRequestContent(
                input_text="hello",
                target_prompt_tokens=1,
            ),
            requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=3)),
        )

    result0 = RequestResult(
        request_id=0,
        session_id=0,
        session_total_requests=1,
        channels={
            ChannelModality.TEXT: ChannelResponse(
                modality=ChannelModality.TEXT,
                content="xxx",
                metrics={
                    "is_stream": True,
                    "inter_chunk_times": [0.1, 0.1, 0.1],
                    "num_delta_prompt_tokens": 1,
                    "num_total_prompt_tokens": 1,
                    "num_output_tokens": 3,
                },
            )
        },
        success=True,
        client_completed_at=0.3,
        client_picked_up_at=0.0,
        scheduler_ready_at=0.0,
        scheduler_dispatched_at=0.0,
        result_processed_at=0.3,
    )

    result1 = RequestResult(
        request_id=1,
        session_id=0,
        session_total_requests=1,
        channels={
            ChannelModality.TEXT: ChannelResponse(
                modality=ChannelModality.TEXT,
                content="yyy",
                metrics={
                    "is_stream": True,
                    "inter_chunk_times": [0.15, 0.1, 0.1],
                    "num_delta_prompt_tokens": 1,
                    "num_total_prompt_tokens": 1,
                    "num_output_tokens": 3,
                },
            )
        },
        success=True,
        client_completed_at=0.35,
        client_picked_up_at=0.0,
        scheduler_ready_at=0.0,
        scheduler_dispatched_at=0.0,
        result_processed_at=0.35,
    )

    evaluator.record_request_completed(
        request_id=0, session_id=0, completed_at=0.3, response=result0
    )
    evaluator.record_request_completed(
        request_id=1, session_id=0, completed_at=0.35, response=result1
    )

    evaluator._maybe_save_decode_window_metrics(str(tmp_path))

    out_path = tmp_path / "decode_window_metrics.json"
    assert out_path.exists()

    data = json.loads(out_path.read_text())
    assert data["window"]["start"] == pytest.approx(0.15)
    assert data["window"]["end"] == pytest.approx(0.3)
    assert data["tbc_in_window_stats"]["count"] == 3
    assert data["tbc_in_window_stats"]["mean"] == pytest.approx(0.1)





