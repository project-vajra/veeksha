import pytest

from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.evaluator.performance.text import TextPerformanceEvaluator
from veeksha.types import ChannelModality


@pytest.mark.unit
def test_text_performance_skips_tpot_for_non_stream_request() -> None:
    config = PerformanceEvaluatorConfig(stream_metrics=False)
    evaluator = TextPerformanceEvaluator(
        config=config,
        channel_config=config.text_channel,
        benchmark_start_time=0.0,
    )

    evaluator.register_request(
        request_id=0,
        session_id=0,
        dispatched_at=0.0,
        content=TextChannelRequestContent(
            input_text="hello",
            target_prompt_tokens=1,
        ),
        requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=1)),
    )

    result = RequestResult(
        request_id=0,
        session_id=0,
        session_total_requests=1,
        channels={
            ChannelModality.TEXT: ChannelResponse(
                modality=ChannelModality.TEXT,
                content="x",
                metrics={
                        "is_stream": False,
                    # Non-stream: only one duration for the whole response
                    "inter_chunk_times": [0.5],
                    "num_delta_prompt_tokens": 1,
                    "num_total_prompt_tokens": 1,
                    "num_output_tokens": 1,
                },
            )
        },
        success=True,
        client_completed_at=0.5,
    )

    evaluator.record_request_completed(
        request_id=0,
        session_id=0,
        completed_at=0.5,
        response=result,
    )

    assert len(evaluator.summaries["ttfc"]) == 1
    assert len(evaluator.summaries["tpot"]) == 0


@pytest.mark.unit
def test_text_performance_includes_tpot_for_streaming_request() -> None:
    config = PerformanceEvaluatorConfig(stream_metrics=False)
    evaluator = TextPerformanceEvaluator(
        config=config,
        channel_config=config.text_channel,
        benchmark_start_time=0.0,
    )

    evaluator.register_request(
        request_id=0,
        session_id=0,
        dispatched_at=0.0,
        content=TextChannelRequestContent(
            input_text="hello",
            target_prompt_tokens=1,
        ),
        requested_output=RequestedOutputSpec(text=TextOutputSpec(target_tokens=2)),
    )

    result = RequestResult(
        request_id=0,
        session_id=0,
        session_total_requests=1,
        channels={
            ChannelModality.TEXT: ChannelResponse(
                modality=ChannelModality.TEXT,
                content="xy",
                metrics={
                    "is_stream": True,
                    # Streaming: at least TTFC + one chunk gap
                    "inter_chunk_times": [0.1, 0.2],
                    "num_delta_prompt_tokens": 1,
                    "num_total_prompt_tokens": 1,
                    "num_output_tokens": 2,
                },
            )
        },
        success=True,
        client_completed_at=0.3,
    )

    evaluator.record_request_completed(
        request_id=0,
        session_id=0,
        completed_at=0.3,
        response=result,
    )

    assert len(evaluator.summaries["tpot"]) == 1
