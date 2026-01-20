"""Unit tests for TextPerformanceEvaluator."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pytest

from veeksha.config.evaluator import (
    PerformanceEvaluatorConfig,
    TextChannelPerformanceConfig,
)
from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
from veeksha.evaluator.performance.text import TextPerformanceEvaluator
from veeksha.types import ChannelModality


@dataclass
class MockChannelResponseContent:
    metrics: Dict[str, Any]


@dataclass
class MockResponse:
    channels: Dict[ChannelModality, MockChannelResponseContent]
    session_total_requests: int = 1
    scheduler_ready_at: Optional[float] = None
    scheduler_dispatched_at: Optional[float] = None
    client_picked_up_at: Optional[float] = None
    client_completed_at: Optional[float] = None
    result_processed_at: Optional[float] = None


@dataclass
class MockRequestContent:
    target_prompt_tokens: int = 5


@pytest.fixture
def evaluator() -> TextPerformanceEvaluator:
    config = PerformanceEvaluatorConfig()
    channel_config = TextChannelPerformanceConfig()
    return TextPerformanceEvaluator(config, channel_config)


@pytest.mark.unit
def test_end_to_end_metrics(evaluator: TextPerformanceEvaluator) -> None:
    request_id = 1
    session_id = 1
    dispatched_at = 10.0
    completed_at = 12.0
    
    # 1. Register request with requested_output
    content = MockRequestContent()
    requested_output = RequestedOutputSpec(text=TextOutputSpec(target_tokens=10))
    evaluator.register_request(request_id, session_id, dispatched_at, content, requested_output)
    
    assert request_id in evaluator._pending_requests
    
    # 2. Complete request
    # Inter-chunk times: [0.5 (TTFC), 0.1, 0.1 ... ]
    # Total 2.0s latency.
    # 10 output tokens.
    inter_chunk_times = [0.5] + [0.15] * 10 
    # Sum = 0.5 + 1.5 = 2.0
    
    metrics = {
        "num_total_prompt_tokens": 20,
        "num_delta_prompt_tokens": 5,
        "num_output_tokens": 11, # 10 chunks after first + first chunk = 11 tokens? 
        # Actually logic is sum(inter_chunk_times) = latency.
        # If len(inter_chunk_times) == num_output_tokens typically.
        "inter_chunk_times": inter_chunk_times,
        "is_stream": True,
    }
    
    response = MockResponse(
        channels={ChannelModality.TEXT: MockChannelResponseContent(metrics)},
        client_completed_at=completed_at
    )
    
    evaluator.record_request_completed(request_id, session_id, completed_at, response)
    
    assert request_id not in evaluator._pending_requests
    
    # Check metrics
    idx = evaluator.request_ids.index(request_id)
    assert evaluator.ttfc[idx] == 0.5
    assert evaluator.num_output_tokens[idx] == 11
    
    # TPOT: (E2E - TTFC) / (OutputTokens - 1)
    # E2E = sum(0.5 + 1.5) = 2.0
    # TPOT = (2.0 - 0.5) / (11 - 1) = 1.5 / 10 = 0.15
    assert evaluator.tpot[idx] == pytest.approx(0.15)
    
    assert evaluator.end_to_end_latency[idx] == pytest.approx(2.0)


@pytest.mark.unit
def test_non_streaming_metrics(evaluator: TextPerformanceEvaluator) -> None:
    request_id = 2
    session_id = 1
    dispatched_at = 20.0
    completed_at = 21.0
    
    evaluator.register_request(request_id, session_id, dispatched_at, MockRequestContent())
    
    metrics = {
        "num_output_tokens": 10,
        "inter_chunk_times": [1.0], # Single chunk for non-streaming
        "is_stream": False,
    }
    
    response = MockResponse(
        channels={ChannelModality.TEXT: MockChannelResponseContent(metrics)},
        client_completed_at=completed_at
    )
    
    evaluator.record_request_completed(request_id, session_id, completed_at, response)
    
    idx = evaluator.request_ids.index(request_id)
    assert not evaluator.is_stream[idx]
    assert evaluator.ttfc[idx] == 1.0


@pytest.mark.unit
def test_session_think_time(evaluator: TextPerformanceEvaluator) -> None:
    session_id = 100
    
    # Request 1
    evaluator.register_request(101, session_id, 10.0, MockRequestContent())
    evaluator.record_request_completed(
        101, session_id, 11.0, 
        MockResponse(channels={ChannelModality.TEXT: MockChannelResponseContent({})})
    )
    
    # Request 2 (dispatched at 13.0, completed at 11.0)
    # Think time = 13.0 - 11.0 = 2.0
    evaluator.register_request(102, session_id, 13.0, MockRequestContent())
    evaluator.record_request_completed(
        102, session_id, 14.0, 
        MockResponse(channels={ChannelModality.TEXT: MockChannelResponseContent({})})
    )
    
    summary = evaluator.get_summary()
    # Check if think time has been recorded.
    val = summary.get("Intra-session Think Time (Mean)")
    assert val is not None
    assert val == 2.0
