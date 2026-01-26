"""Unit tests for ImagePerformanceEvaluator."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from veeksha.config.evaluator import (
    ImageChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.core.requested_output import ImageOutputSpec, RequestedOutputSpec
from veeksha.evaluator.performance.image import ImagePerformanceEvaluator
from veeksha.types import ChannelModality


@dataclass
class MockChannelResponseContent:
    metrics: Dict[str, Any]
    content: list = None

    def __post_init__(self):
        if self.content is None:
            self.content = []


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
    pass


@pytest.fixture
def evaluator() -> ImagePerformanceEvaluator:
    config = PerformanceEvaluatorConfig()
    channel_config = ImageChannelPerformanceConfig()
    return ImagePerformanceEvaluator(config, channel_config)


@pytest.mark.unit
def test_non_streaming_metrics(evaluator: ImagePerformanceEvaluator) -> None:
    """Test non-streaming image generation metrics."""
    request_id = 1
    session_id = 1
    dispatched_at = 10.0
    completed_at = 12.0
    
    # 1. Register request with requested_output
    requested_output = RequestedOutputSpec(image=ImageOutputSpec(num_images=2, size="1024x1024"))
    evaluator.register_request(request_id, session_id, dispatched_at, MockRequestContent(), requested_output)
    
    assert request_id in evaluator._pending_requests
    
    # 2. Complete request
    # Non-streaming: single chunk time represents total E2E latency
    inter_chunk_times = [2.0]  # Total 2.0s to generate images
    
    metrics = {
        "num_total_prompt_tokens": 15,
        "num_delta_prompt_tokens": 15,
        "num_output_images": 2,
        "inter_chunk_times": inter_chunk_times,
        "is_stream": False,
    }
    
    response = MockResponse(
        channels={ChannelModality.IMAGE: MockChannelResponseContent(metrics, content=[b"img1", b"img2"])},
        client_completed_at=completed_at
    )
    
    evaluator.record_request_completed(request_id, session_id, completed_at, response)
    
    assert request_id not in evaluator._pending_requests
    
    # Check metrics
    idx = evaluator.request_ids.index(request_id)
    assert evaluator.num_generated_images[idx] == 2
    assert evaluator.num_prompt_tokens[idx] == 15
    assert evaluator.end_to_end_latency[idx] == pytest.approx(2.0)
    
    # Latency per image = 2.0 / 2 = 1.0s per image
    assert evaluator.latency_per_image[idx] == pytest.approx(1.0)
    
    # Generation rate = 2 images / 2.0s = 1.0 images/s
    assert evaluator.generation_rate[idx] == pytest.approx(1.0)
    
    # Check is_stream flag
    assert evaluator.is_stream[idx] is False
    
    # Check images were stored
    assert request_id in evaluator.images
    assert len(evaluator.images[request_id]) == 2


@pytest.mark.unit
def test_session_think_time(evaluator: ImagePerformanceEvaluator) -> None:
    """Test session think time calculation between requests."""
    session_id = 100
    
    # Request 1: dispatched at 10.0, completed at 12.0
    evaluator.register_request(101, session_id, 10.0, MockRequestContent())
    
    metrics_1 = {
        "num_total_prompt_tokens": 10,
        "num_output_images": 1,
        "inter_chunk_times": [2.0],
        "is_stream": False,
    }
    
    response_1 = MockResponse(
        channels={ChannelModality.IMAGE: MockChannelResponseContent(metrics_1, content=[b"img1"])},
        client_completed_at=12.0
    )
    
    evaluator.record_request_completed(101, session_id, 12.0, response_1)
    
    # Request 2: dispatched at 15.0, completed at 17.0
    # Think time = 15.0 - 12.0 = 3.0
    evaluator.register_request(102, session_id, 15.0, MockRequestContent())
    
    metrics_2 = {
        "num_total_prompt_tokens": 10,
        "num_output_images": 1,
        "inter_chunk_times": [2.0],
        "is_stream": False,
    }
    
    response_2 = MockResponse(
        channels={ChannelModality.IMAGE: MockChannelResponseContent(metrics_2, content=[b"img2"])},
        client_completed_at=17.0
    )
    
    evaluator.record_request_completed(102, session_id, 17.0, response_2)
    
    # Check that think time was recorded
    summary = evaluator.get_summary()
    think_time_mean = summary.get("Session think time (Mean)")
    assert think_time_mean is not None
    assert think_time_mean == pytest.approx(3.0)
