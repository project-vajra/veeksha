from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from veeksha.config.evaluator import BaseEvaluatorConfig
from veeksha.core.seeding import SeedManager
from veeksha.types import ChannelModality


@dataclass
class EvaluationResult:
    """Container for evaluation results from any evaluator.

    Attributes:
        evaluator_type: String identifying the evaluator type (e.g., "performance", "accuracy")
        channel: Optional channel modality this result applies to (None for aggregate results)
        metrics: Dictionary of computed metrics
        raw_data: Optional dictionary containing detailed per-request data
    """

    evaluator_type: str
    channel: Optional[ChannelModality]
    metrics: Dict[str, Any]
    raw_data: Optional[Dict[str, Any]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator_type": self.evaluator_type,
            "channel": str(self.channel) if self.channel else None,
            "metrics": self.metrics,
        }


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators.

    Evaluators consume completed requests/responses and produce evaluation metrics.
    They might operate on specific channels or aggregate across all channels.

    The lifecycle would be:
    1. Construction with config
    2. register_request() called for each request when dispatched
    3. record_request_completed() called when response received
    4. record_session_completed() called when session finishes
    5. finalize() called at the end to compute aggregate metrics
    6. save() called to persist results to disk
    """

    def __init__(
        self,
        config: BaseEvaluatorConfig,
        seed_manager: Optional[SeedManager] = None,
    ):
        """Initialize the evaluator.

        Args:
            config: Evaluator configuration
            seed_manager: Optional seed manager for reproducibility
        """
        self.config = config
        self.seed_manager = seed_manager
        self._target_channels = (
            set(config.target_channels) if config.target_channels else None
        )

    def should_evaluate_channel(self, channel: ChannelModality) -> bool:
        """Check if this evaluator should process a given channel.

        Args:
            channel: The channel modality to check
        """
        if self._target_channels is None:
            return True  # all channels
        return channel in self._target_channels

    @abstractmethod
    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        channels: Dict[ChannelModality, Any],
        requested_output: Any = None,
    ) -> None:
        """Register a request that was dispatched to the server

        Args:
            request_id: Unique request id
            session_id: Session id this request belongs to
            dispatched_at: Monotonic timestamp when request was dispatched
            channels: Request content indexed by modality
            requested_output: Optional output specification for the request
        """
        raise NotImplementedError

    @abstractmethod
    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
        error: Optional[Exception] = None,
    ) -> None:
        """Record that a request completed.

        Args:
            request_id: Unique request id
            session_id: Session id this request belongs to
            completed_at: Monotonic timestamp when response was received
            response: Response object containing timing and content data
            error: Optional exception if the request failed
        """
        raise NotImplementedError

    @abstractmethod
    def record_session_completed(
        self,
        session_id: int,
        completed_at: float,
        success: bool,
    ) -> None:
        """Record that a session completed.

        Args:
            session_id: Session id
            completed_at: Monotonic timestamp when session completed
            success: True if all requests in session succeeded
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> EvaluationResult:
        """Finalize evaluation and return results.

        Returns:
            EvaluationResult containing computed metrics
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, output_dir: str) -> None:
        """Save evaluation artifacts to the output directory.

        Args:
            output_dir: Directory to write output files to
        """
        raise NotImplementedError

    def get_streaming_metrics(self) -> Optional[Dict[str, Any]]:
        """Return current metrics for streaming updates.

        Returns:
            Dictionary of current metrics, or None if streaming not supported
        """
        return None

    def get_completed_request_count(self) -> int:
        """Return count of completed requests for progress tracking.

        Returns:
            Number of completed requests. Subclasses should override.
        """
        return 0

    def get_session_counts(self) -> tuple:
        """Return session counts for progress tracking.

        Returns:
            Tuple of (completed_sessions, errored_sessions, in_progress_sessions).
            Subclasses should override.
        """
        return (0, 0, 0)

    def set_included_requests(self, request_ids: Set[int]) -> None:
        """Set which requests to include in final metrics.

        If not called, all requests are included. Subclasses can override
        to filter metrics during finalization.

        Args:
            request_ids: Set of request IDs to include in metrics
        """
