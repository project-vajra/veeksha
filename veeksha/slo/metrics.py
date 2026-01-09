"""Metric extraction helpers for SLO evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def lower_is_better(metric: str) -> bool:
    """Return whether lower values are better for the given metric."""
    # Today we only support latency-like metrics.
    return True


def extract_metric_values(
    metric: str, request_level_metrics: Dict[str, Any]
) -> List[float]:
    """Extract numeric samples for a metric from request-level metrics.

    Args:
        metric: Metric key. Currently supported: "ttfc", "tbc".
        request_level_metrics: Dict of metric -> list of values (decoded from JSONL).

    Returns:
        Flat list of float samples for percentile evaluation.
    """
    # Map SLO metric names to request-level metric keys
    key = metric
    if metric == "e2e":
        key = "end_to_end_latency"
    elif metric == "tpot":
        key = "tpot"

    raw = request_level_metrics.get(key, [])
    if not isinstance(raw, list) or not raw:
        return []

    if metric == "tbc":
        # values per-request as List[List[float]] (one list per request).
        # flatten to a single series of chunk gaps.
        if raw and isinstance(raw[0], list):
            return _flatten_float_lists(raw)

    return _coerce_float_list(raw)


def _flatten_float_lists(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for sublist in values:
        if isinstance(sublist, list):
            out.extend(_coerce_float_list(sublist))
        else:
            item = _coerce_float(sublist)
            if item is not None:
                out.append(item)
    return out


def _coerce_float_list(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        item = _coerce_float(v)
        if item is not None:
            out.append(item)
    return out


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
