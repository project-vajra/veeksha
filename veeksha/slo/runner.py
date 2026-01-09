from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence, Set

from veeksha.config.slo import BaseSloConfig
from veeksha.logger import init_logger
from veeksha.slo.evaluator import SloEvaluationResult, SloEvaluator
from veeksha.slo.slo import SloSet

logger = init_logger(__name__)


def load_request_level_metrics_jsonl(
    path: str, *, keys: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """Load request-level metrics JSONL into a dict-of-lists.

    Args:
        path: Path to request_level_metrics.jsonl
        keys: If provided, only these keys are loaded (reduces memory/CPU).

    Returns:
        Dict mapping metric keys to list-of-values.
    """
    metrics: Dict[str, Any] = {}
    if not os.path.exists(path):
        return metrics

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            if keys is None:
                for key, value in row.items():
                    metrics.setdefault(key, []).append(value)
            else:
                for key in keys:
                    if key in row:
                        metrics.setdefault(key, []).append(row[key])
    return metrics


def evaluate_and_save_slos(
    slo_configs: Sequence[BaseSloConfig],
    metrics_dir: str,
) -> Optional[SloEvaluationResult]:
    """Evaluate configured SLOs and persist results to the metrics directory.

    Persists:
    - `slo_results.json`: machine-readable summary for meta-benchmarks

    Args:
        slo_configs: SLO configs to evaluate.
        metrics_dir: Directory containing request-level metrics and where SLO
            results should be written.

    Returns:
        SloEvaluationResult if SLOs were evaluated, else None.
    """
    if not slo_configs:
        return None

    os.makedirs(metrics_dir, exist_ok=True)

    request_metrics_path = os.path.join(metrics_dir, "request_level_metrics.jsonl")
    request_level_metrics = load_request_level_metrics_jsonl(
        request_metrics_path, keys=_get_slo_metric_keys(slo_configs)
    )
    if not request_level_metrics:
        logger.warning(
            "SLO evaluation requested but request-level metrics file is missing/empty: %s",
            request_metrics_path,
        )

    slo_set = SloSet(list(slo_configs))
    evaluator = SloEvaluator(slo_set)
    evaluation_result = evaluator.evaluate(request_level_metrics)

    out_path = os.path.join(metrics_dir, "slo_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_result.to_dict(), f, indent=2)

    return evaluation_result


def _get_slo_metric_keys(slo_configs: Sequence[BaseSloConfig]) -> Set[str]:
    keys: Set[str] = set()
    for slo in slo_configs:
        metric = getattr(slo, "metric", None)
        if isinstance(metric, str):
            if metric == "e2e":
                keys.add("end_to_end_latency")
            else:
                keys.add(metric)
    return keys
