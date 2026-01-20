"""Composite evaluator for running multiple evaluators in a single benchmark."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

from veeksha.evaluator.base import BaseEvaluator, EvaluationResult


class CompositeEvaluator(BaseEvaluator):
    """Fan-out wrapper that runs multiple evaluators over the same run."""

    def __init__(self, evaluators: list[BaseEvaluator]):
        if not evaluators:
            raise ValueError("CompositeEvaluator requires at least one evaluator.")
        super().__init__(
            config=evaluators[0].config, seed_manager=evaluators[0].seed_manager
        )
        self._evaluators = evaluators

    @property
    def _primary(self) -> BaseEvaluator:
        """Evaluator used for progress tracking and timeout filtering."""
        return self._evaluators[0]

    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        channels: Dict[Any, Any],
        requested_output: Any = None,
    ) -> None:
        for evaluator in self._evaluators:
            evaluator.register_request(
                request_id=request_id,
                session_id=session_id,
                dispatched_at=dispatched_at,
                channels=channels,
                requested_output=requested_output,
            )

    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
        error: Optional[Exception] = None,
    ) -> None:
        for evaluator in self._evaluators:
            evaluator.record_request_completed(
                request_id=request_id,
                session_id=session_id,
                completed_at=completed_at,
                response=response,
                error=error,
            )

    def record_session_completed(
        self,
        session_id: int,
        completed_at: float,
        success: bool,
    ) -> None:
        for evaluator in self._evaluators:
            evaluator.record_session_completed(
                session_id=session_id, completed_at=completed_at, success=success
            )

    def finalize(self) -> EvaluationResult:
        results = [e.finalize() for e in self._evaluators]

        metrics: Dict[str, Any] = {}
        raw: Dict[str, Any] = {"evaluators": []}
        for idx, result in enumerate(results):
            key = result.evaluator_type or f"evaluator_{idx}"
            # Avoid collisions if evaluator_type repeats.
            if key in metrics:
                key = f"{key}_{idx}"
            metrics[key] = result.metrics
            raw["evaluators"].append(result.to_dict())

        return EvaluationResult(
            evaluator_type="composite",
            channel=None,
            metrics=metrics,
            raw_data=raw,
        )

    def save(self, output_dir: str) -> None:
        for evaluator in self._evaluators:
            evaluator.save(output_dir)

    # ---------------------------------------------------------------------
    # Progress / timeout hooks
    # ---------------------------------------------------------------------

    def get_streaming_metrics(self) -> Optional[Dict[str, Any]]:
        return self._primary.get_streaming_metrics()

    def get_completed_request_count(self) -> int:
        return self._primary.get_completed_request_count()

    def get_session_counts(self) -> tuple[int, int, int]:
        return self._primary.get_session_counts()

    def set_included_requests(self, request_ids: Set[int]) -> None:
        for evaluator in self._evaluators:
            evaluator.set_included_requests(request_ids)

    def get_registered_request_ids(self) -> Set[int]:
        primary = self._primary
        if hasattr(primary, "get_registered_request_ids"):
            return primary.get_registered_request_ids()  # type: ignore[no-any-return]
        return set()
