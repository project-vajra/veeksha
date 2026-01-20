"""Accuracy evaluators (new veeksha framework).

Currently the only supported accuracy backend is `lm-eval`. We keep the lm-eval
implementation in a dedicated evaluator class so we can add additional accuracy
backends without conflating their requirements (e.g., generator coupling,
response parsing, artifact formats).
"""

from __future__ import annotations

import json
import os
import threading
from abc import ABC
from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple, cast

# NOTE: `lm_eval` is an external dependency; type checkers in some environments
# may not have it available, so we silence missing-import diagnostics here.
from lm_eval.evaluator_utils import (  # pyright: ignore[reportMissingImports]  # type: ignore[import-not-found]
    consolidate_group_results,
    consolidate_results,
    get_subtask_list,
    prepare_print_tasks,
)

from veeksha.config.evaluator import LMEvalAccuracyEvaluatorConfig
from veeksha.core.seeding import SeedManager
from veeksha.evaluator.base import BaseEvaluator, EvaluationResult
from veeksha.generator.session.lmeval import LMEvalSessionGenerator
from veeksha.logger import init_logger
from veeksha.types import ChannelModality, LMEvalOutputType

logger = init_logger(__name__)


class BaseAccuracyEvaluator(BaseEvaluator, ABC):
    """Base class for accuracy evaluators.

    Accuracy evaluators are allowed to be workload-generator-specific (e.g. lm-eval
    binds responses back to generator-owned task instances). Concrete subclasses
    should document any required coupling explicitly.
    """

    def __init__(
        self,
        config: LMEvalAccuracyEvaluatorConfig,
        seed_manager: Optional[SeedManager] = None,
        output_dir: Optional[str] = None,
        benchmark_start_time: float = 0.0,
    ):
        super().__init__(config=config, seed_manager=seed_manager)
        self.config = config
        self.output_dir = output_dir
        self.benchmark_start_time = benchmark_start_time


class LMEvalAccuracyEvaluator(BaseAccuracyEvaluator):
    """Accuracy evaluator that computes lm-eval metrics from completed requests."""

    def __init__(
        self,
        config: LMEvalAccuracyEvaluatorConfig,
        seed_manager: Optional[SeedManager] = None,
        output_dir: Optional[str] = None,
        benchmark_start_time: float = 0.0,
        session_generator: Optional[Any] = None,
    ):
        super().__init__(
            config=config,
            seed_manager=seed_manager,
            output_dir=output_dir,
            benchmark_start_time=benchmark_start_time,
        )

        if not isinstance(session_generator, LMEvalSessionGenerator):
            raise ValueError(
                "LMEvalAccuracyEvaluator requires LMEvalSessionGenerator via session_generator=..."
            )
        self.session_generator: LMEvalSessionGenerator = session_generator

        self._lock = threading.Lock()

        self._registered_request_ids: Set[int] = set()
        self._included_request_ids: Optional[Set[int]] = None

        self.num_requests: int = 0
        self.num_completed_requests: int = 0
        self.num_errored_requests: int = 0

        self._results_dict: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------------
    # Request lifecycle
    # ---------------------------------------------------------------------

    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        channels: Dict[ChannelModality, Any],
        requested_output: Any = None,
    ) -> None:
        with self._lock:
            self.num_requests += 1
            self._registered_request_ids.add(request_id)

    def get_registered_request_ids(self) -> Set[int]:
        with self._lock:
            return set(self._registered_request_ids)

    def set_included_requests(self, request_ids: Set[int]) -> None:
        with self._lock:
            self._included_request_ids = set(request_ids)
            logger.info(
                "LMEvalAccuracyEvaluator: set included requests to %d", len(request_ids)
            )

    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
        error: Optional[Exception] = None,
    ) -> None:
        with self._lock:
            if (
                self._included_request_ids is not None
                and request_id not in self._included_request_ids
            ):
                return

            if error is not None or not getattr(response, "success", True):
                self.num_errored_requests += 1
                return

            instance = self.session_generator.instance_by_request_id.get(request_id)
            if instance is None:
                self.num_errored_requests += 1
                logger.warning(
                    "LMEvalAccuracyEvaluator: missing lm-eval instance for request_id=%d",
                    request_id,
                )
                return

            text_channel = getattr(response, "channels", {}).get(ChannelModality.TEXT)
            if text_channel is None:
                self.num_errored_requests += 1
                logger.warning(
                    "LMEvalAccuracyEvaluator: missing text channel for request_id=%d",
                    request_id,
                )
                return

            if instance.request_type == str(LMEvalOutputType.GENERATE_UNTIL):
                instance.resps.append(text_channel.content)
                self.num_completed_requests += 1
                return

            if instance.request_type == str(LMEvalOutputType.LOGLIKELIHOOD):
                ctxlen = self.session_generator.ctxlen_tokens_by_request_id.get(
                    request_id
                )
                logprobs = getattr(text_channel, "metrics", {}).get("logprobs")
                if ctxlen is None or logprobs is None:
                    self.num_errored_requests += 1
                    logger.warning(
                        "LMEvalAccuracyEvaluator: missing ctxlen/logprobs for request_id=%d",
                        request_id,
                    )
                    return

                instance.resps.append(self._parse_logprobs(ctxlen=ctxlen, lp=logprobs))
                self.num_completed_requests += 1
                return

            raise NotImplementedError(
                f"lm-eval request_type '{instance.request_type}' not supported."
            )

    def record_session_completed(
        self,
        session_id: int,
        completed_at: float,
        success: bool,
    ) -> None:
        # lm-eval sessions may contain multiple requests; handled via request completion.
        return

    # ---------------------------------------------------------------------
    # lm-eval evaluation
    # ---------------------------------------------------------------------

    def _parse_logprobs(self, ctxlen: int, lp: Any) -> Tuple[float, bool]:
        """Parse per-token logprobs for completions-style responses.

        Supports multiple provider formats (copied from legacy lmeval generator):
        1) OpenAI-compatible dict with keys: token_logprobs, top_logprobs.
        2) content list: {"content": [{"token","logprob","top_logprobs":[...]}...]}
        3) chunks list: {"chunks": [{"logprob" or "token_logprobs":[...], ...}, ...]}
        """
        if not isinstance(lp, dict):
            raise KeyError("Unsupported logprobs structure (expected dict).")

        # Case 1: tokens/token_logprobs arrays
        if "token_logprobs" in lp and "top_logprobs" in lp:
            tokens_logprobs = lp["token_logprobs"][ctxlen:-1]
            top_logprobs = lp["top_logprobs"][ctxlen:-1]
            logprobs_sum = sum(tokens_logprobs)
            is_greedy = True
            for tok_lp, top in zip(tokens_logprobs, top_logprobs):
                if isinstance(top, dict):
                    if not top:
                        is_greedy = False
                        break
                    eps = 1e-8
                    if tok_lp < (max(top.values()) - eps):
                        is_greedy = False
                        break
                else:
                    is_greedy = False
                    break
            return (logprobs_sum, is_greedy)

        # Case 2: content list with per-token objects (non-stream)
        if isinstance(lp.get("content"), list):
            content = lp["content"]
            sliced = content[ctxlen:]
            logprobs_list: list[float] = []
            greedies: list[bool] = []
            for entry in sliced:
                tok_lp = entry.get("logprob")
                if tok_lp is None:
                    continue
                logprobs_list.append(tok_lp)
                top = entry.get("top_logprobs") or []
                max_top = None
                if isinstance(top, list) and top:
                    try:
                        max_top = max((t.get("logprob", float("-inf")) for t in top))
                    except Exception:
                        max_top = None
                greedies.append(max_top is not None and tok_lp >= max_top)
            logprobs_sum = sum(logprobs_list) if logprobs_list else 0.0
            is_greedy = all(greedies) if greedies else False
            return (logprobs_sum, is_greedy)

        # Case 3: chunks list (streaming-style)
        if isinstance(lp.get("chunks"), list):
            chunks = lp["chunks"]
            chunks_logprobs_list: list[float] = []
            chunks_greedies: list[bool] = []
            for entry in chunks:
                tok_lp = entry.get("logprob")
                if tok_lp is None and isinstance(entry.get("token_logprobs"), list):
                    try:
                        tok_lp = float(entry["token_logprobs"][0])
                    except Exception:
                        tok_lp = None
                if tok_lp is None:
                    continue
                chunks_logprobs_list.append(tok_lp)
                top = entry.get("top_logprobs") or []
                max_top = None
                if isinstance(top, list) and top:
                    try:
                        max_top = max((t.get("logprob", float("-inf")) for t in top))
                    except Exception:
                        max_top = None
                chunks_greedies.append(max_top is not None and tok_lp >= max_top)
            logprobs_sum = sum(chunks_logprobs_list) if chunks_logprobs_list else 0.0
            is_greedy = all(chunks_greedies) if chunks_greedies else False
            return (logprobs_sum, is_greedy)

        raise KeyError("Unsupported logprobs structure for completions response.")

    def _evaluate_lmeval(self) -> Dict[str, Any]:
        # Bind generator-owned tasks and compute lm-eval metrics
        eval_tasks = self.session_generator.eval_tasks
        limits = self.session_generator.limits

        # Ensure idempotence if finalize() is called multiple times.
        for task_output in eval_tasks:
            task_output.sample_metrics.clear()
            task_output.agg_metrics.clear()

        for task_output, limit in zip(eval_tasks, limits):
            task = task_output.task
            if task is None:
                continue
            task.apply_filters()

            instances_by_doc_id = defaultdict(list)
            for instance in task.instances:
                instances_by_doc_id[instance.doc_id].append(instance)
            for instances in instances_by_doc_id.values():
                instances.sort(key=lambda x: x.idx)

            if not task.instances:
                continue

            for filter_key in task.instances[0].filtered_resps.keys():
                doc_iterator = task.doc_iterator(limit=limit)
                for doc_id, doc in doc_iterator:
                    requests = instances_by_doc_id[doc_id]
                    metrics = task.process_results(
                        doc, [req.filtered_resps[filter_key] for req in requests]
                    )
                    for metric, value in metrics.items():  # type: ignore
                        task_output.sample_metrics[(metric, filter_key)].append(value)

        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(
                bootstrap_iters=self.config.bootstrap_iters
            )

        (
            results,
            samples,
            configs,
            versions,
            num_fewshot,
            higher_is_better,
        ) = consolidate_results(eval_tasks)

        show_group_table = False
        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(
                results, versions, self.session_generator.task_dict
            )
        results_agg, group_agg = prepare_print_tasks(
            self.session_generator.task_dict, results
        )
        subtask_list = get_subtask_list(self.session_generator.task_dict)

        # propagate higher_is_better to groups
        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if len(task_list) != 0:
                for task_name in task_list:
                    for metric, hib in higher_is_better[task_name].items():
                        if metric not in _higher_is_better:
                            _higher_is_better[metric] = hib
                        if (
                            metric in _higher_is_better
                            and _higher_is_better[metric] is not None
                            and _higher_is_better[metric] != hib
                        ):
                            _higher_is_better[metric] = None
                higher_is_better[group] = _higher_is_better

        results_dict: Dict[str, Any] = {
            "results": dict(results_agg.items()),
            **(
                {"groups": dict(group_agg.items())}
                if (bool(group_agg) and show_group_table)
                else {}
            ),
            "group_subtasks": dict(reversed(subtask_list.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": {
                task_output.task_name: {
                    "original": len(task.eval_docs),
                    "effective": min(
                        (limit if limit is not None else len(task.eval_docs)),
                        len(task.eval_docs),
                    ),
                }
                for task_output, limit in zip(eval_tasks, limits)
                for task in (cast(Any, task_output.task),)
            },
        }

        return results_dict

    def finalize(self) -> EvaluationResult:
        with self._lock:
            self._results_dict = self._evaluate_lmeval()
            metrics = {
                "num_requests": self.num_requests,
                "num_completed_requests": self.num_completed_requests,
                "num_errored_requests": self.num_errored_requests,
                "num_tasks": (
                    len(self._results_dict.get("results", {}))
                    if self._results_dict
                    else 0
                ),
            }
            return EvaluationResult(
                evaluator_type="accuracy",
                channel=None,
                metrics=metrics,
                raw_data={"lmeval": self._results_dict},
            )

    def save(self, output_dir: str) -> None:
        if self._results_dict is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "lmeval_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._results_dict, f, indent=2)

    # ---------------------------------------------------------------------
    # Progress tracking
    # ---------------------------------------------------------------------

    def get_completed_request_count(self) -> int:
        with self._lock:
            return self.num_completed_requests

    def get_session_counts(self) -> tuple[int, int, int]:
        with self._lock:
            completed = self.num_completed_requests
            errored = self.num_errored_requests
            in_progress = max(
                0, len(self._registered_request_ids) - completed - errored
            )
            return completed, errored, in_progress
