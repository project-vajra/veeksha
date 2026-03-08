"""Shared utilities for microbenchmark modules."""

import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

from rich.console import Console

from veeksha.config.client import OpenAIChatCompletionsClientConfig
from veeksha.logger import init_logger
from veeksha.microbench.config import BaseMicrobenchmarkConfig

logger = init_logger(__name__)
console = Console()

_OUTPUT_TOKEN_MULTIPLIER = 2


# ---------------------------------------------------------------------------
# Client config
# ---------------------------------------------------------------------------


def build_client_config(
    cfg: BaseMicrobenchmarkConfig,
) -> OpenAIChatCompletionsClientConfig:
    return OpenAIChatCompletionsClientConfig(
        model=cfg.model,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        request_timeout=cfg.request_timeout,
        max_tokens_param=cfg.max_tokens_param,
        ignore_eos=cfg.ignore_eos,
    )


# ---------------------------------------------------------------------------
# Engine math
# ---------------------------------------------------------------------------


def compute_prefill_iterations(
    input_length: int, chunk_size: int, active_decodes: int
) -> int:
    """Iterations needed to prefill one request given active decode slots.

    Each iteration has a token budget of *chunk_size*.  Active decode
    requests consume one token each, leaving the rest for prefill.
    """
    effective_chunk = chunk_size - active_decodes
    assert (
        effective_chunk > 0
    ), f"chunk_size ({chunk_size}) must exceed active_decodes ({active_decodes})"
    return math.ceil(input_length / effective_chunk)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class ValidationResult:
    """Accumulates pass/warn/fail checks."""

    def __init__(self) -> None:
        self.checks: list[tuple[str, str, str]] = []  # (status, name, detail)

    def passed(self, name: str, detail: str = "") -> None:
        self.checks.append(("PASS", name, detail))

    def warn(self, name: str, detail: str) -> None:
        self.checks.append(("WARN", name, detail))

    def fail(self, name: str, detail: str) -> None:
        self.checks.append(("FAIL", name, detail))

    @property
    def ok(self) -> bool:
        return all(status != "FAIL" for status, _, _ in self.checks)

    def summary(self) -> str:
        lines = []
        for status, name, detail in self.checks:
            label = {"PASS": "  PASS", "WARN": "  WARN", "FAIL": "**FAIL"}[status]
            line = f"{label}  {name}"
            if detail:
                line += f" — {detail}"
            lines.append(line)
        num_passed = sum(1 for s, _, _ in self.checks if s == "PASS")
        num_warnings = sum(1 for s, _, _ in self.checks if s == "WARN")
        num_failures = sum(1 for s, _, _ in self.checks if s == "FAIL")
        lines.append(
            f"\n{num_passed} passed, {num_warnings} warnings, {num_failures} failures"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics I/O
# ---------------------------------------------------------------------------


def load_request_metrics(output_dir: str) -> list[dict] | None:
    """Find and load request_level_metrics.jsonl from a benchmark output dir."""
    base = Path(output_dir)
    candidates = sorted(base.glob("**/request_level_metrics.jsonl"))
    if not candidates:
        return None
    path = candidates[-1]
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_decode_window_json(output_dir: str) -> dict | None:
    """Find and load decode_window_metrics.json from a benchmark output dir."""
    base = Path(output_dir)
    candidates = sorted(base.glob("**/decode_window_metrics.json"))
    if not candidates:
        return None
    path = candidates[-1]
    with open(path) as f:
        return json.load(f)


def find_all_run_metrics(base_dir: str) -> list[tuple[Path, list[dict]]]:
    """Find all request_level_metrics.jsonl files under a directory.

    Returns list of (metrics_dir_path, records) sorted by path.
    """
    base = Path(base_dir)
    if not base.exists():
        return []
    results = []
    for path in sorted(base.glob("**/request_level_metrics.jsonl")):
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        if records:
            results.append((path.parent, records))
    return results


def load_decode_window_stats(metrics_dir: Path) -> dict | None:
    """Load decode_window_metrics.json from a metrics directory."""
    path = metrics_dir / "decode_window_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("tbc_in_window_stats")


def save_json(data: Any, path: str) -> None:
    """Save data as JSON, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def fmt_ms(val: float | None) -> str:
    """Format a latency value in seconds as milliseconds."""
    if val is None:
        return "—"
    return f"{val * 1000:.2f}"


def percentile(data: list[float], pct: float) -> float:
    """Compute a percentile from sorted data (0–100 scale)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    frac = (len(sorted_data) - 1) * (pct / 100)
    lo = int(frac)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[lo] + (frac - lo) * (sorted_data[hi] - sorted_data[lo])


def compute_stats(values: list[float]) -> dict[str, Any]:
    """Compute summary statistics for a list of values (in seconds)."""
    return {
        "mean": statistics.mean(values),
        "median": percentile(values, 50),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }
