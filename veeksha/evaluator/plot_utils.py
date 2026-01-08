from __future__ import annotations

"""Plotting utilities for deciding and applying axis scale transforms.

This module provides a minimal, reusable heuristic to choose a numeric axis scale
and small helpers to transform values and format labels accordingly.
"""

from typing import Iterable, Literal

import numpy as np

AxisScale = Literal["linear", "log", "symlog"]
AxisChoice = Literal["x", "y"]


def _finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr)
    return arr[mask]


def recommend_axis_scale(
    values: Iterable[float],
    *,
    kind: Literal["numeric", "probability"] = "numeric",
    threshold: float = 50.0,
    min_samples: int = 10,
) -> AxisScale:
    """Recommend an axis scale for numeric plotting.

    Args:
        values: Sequence of numeric values to be plotted along the axis.
        kind: Treat data as 'numeric' or 'probability'. Probability is always linear.
        threshold: Robust dynamic range (p95/p5) needed to switch to log/symlog.
        min_samples: Minimum number of finite samples required to make a decision.

    Returns:
        One of 'linear', 'log', 'symlog'.
    """
    arr = _finite_array(values)
    if arr.size < min_samples:
        return "linear"

    if kind == "probability":
        return "linear"

    # If all values are within [0, 1], assume probability-like and keep linear.
    if arr.size > 0 and np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0:
        return "linear"

    has_non_positive = np.any(arr <= 0)
    positives = arr[arr > 0]

    if positives.size == 0:
        # No positive mass; only symlog can reasonably represent wide ranges.
        return "symlog"

    q5, q95 = np.quantile(positives, [0.05, 0.95])
    q5 = max(q5, 1e-12)
    robust_ratio = float(q95 / q5) if q5 > 0 else float("inf")

    if robust_ratio >= threshold:
        return "symlog" if has_non_positive else "log"

    return "linear"


def format_axis_label(base: str, unit: str | None, scale: AxisScale) -> str:
    """Format axis label with unit and scale indicator.

    Example:
        format_axis_label("TTFT", "s", "log") -> "TTFT (s, log scale)"
    """
    if scale == "linear":
        return f"{base}{f' ({unit})' if unit else ''}"
    suffix = "log scale" if scale == "log" else "symlog"
    if unit:
        return f"{base} ({unit}, {suffix})"
    return f"{base} ({suffix})"


def apply_axis_scale(
    fig,
    axis: AxisChoice,
    scale: AxisScale,
    *,
    symlog_linthresh: float | None = None,
) -> None:
    """Attempt to set matplotlib axis scale on a Rekha figure.

    Best-effort: tries common attributes to access the underlying matplotlib Axes.
    """
    if scale == "linear":
        return
    try:
        # Preferred: Rekha exposes underlying matplotlib axes via get_axes()
        ax = fig.get_axes()[0]

        if scale == "log":
            if axis == "x":
                ax.set_xscale("log")
            else:
                ax.set_yscale("log")
            return

        # symlog
        linthresh = 1e-6 if symlog_linthresh is None else float(symlog_linthresh)
        if axis == "x":
            ax.set_xscale("symlog", linthresh=linthresh)
        else:
            ax.set_yscale("symlog", linthresh=linthresh)
    except Exception:
        # plots will remain linear.
        return
