from typing import Any, Dict, Optional

from veeksha.config.evaluator import (
    AudioChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.evaluator.base import EvaluationResult
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

logger = init_logger(__name__)


class AudioPerformanceEvaluator:
    """Performance evaluator for audio generation (skeleton).

    TODO: Implement audio-specific metrics
    """

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        channel_config: Optional[AudioChannelPerformanceConfig] = None,
    ):
        self.config = config
        self.channel_config = channel_config or AudioChannelPerformanceConfig()
        logger.warning(
            "AudioPerformanceEvaluator is a skeleton implementation. "
            "Audio metrics are not yet supported."
        )

    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        content: Any,
        requested_output: Any = None,
    ) -> None:
        """Register an audio request that was dispatched."""
        pass  # Skeleton

    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
    ) -> None:
        """Record that an audio request completed."""
        pass  # Skeleton

    def record_session_completed(
        self,
        session_id: int,
        session_size: int,
        first_dispatch_at: Optional[float],
        last_completion_at: Optional[float],
    ) -> None:
        """Record session-level metrics."""
        pass  # Skeleton

    def finalize(self) -> EvaluationResult:
        """Finalize evaluation and return results."""
        return EvaluationResult(
            evaluator_type="audio_performance",
            channel=ChannelModality.AUDIO,
            metrics={
                "status": "not_implemented",
                "message": "Audio performance evaluation not yet implemented",
            },
        )

    def get_streaming_metrics(self) -> Optional[Dict[str, Any]]:
        """Return current metrics for streaming."""
        return None

    def save(self, output_dir: str) -> None:
        """Save evaluation artifacts."""
        pass  # Skeleton

    def flush_streaming_outputs(self, output_dir: str) -> None:
        """Flush current metrics for streaming."""
        pass  # Skeleton
