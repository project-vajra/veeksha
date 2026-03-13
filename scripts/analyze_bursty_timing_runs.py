#!/usr/bin/env python3
"""Compare Veeksha outputs for the bursty-timing experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare short-gap and long-gap Veeksha runs for the bursty timing "
            "experiment."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("traces/bursty_timing/experiment2_metadata.json"),
        help="Metadata file emitted by generate_bursty_timing_trace.py.",
    )
    parser.add_argument(
        "--short-run",
        type=Path,
        required=True,
        help="Veeksha output directory for the short-gap run.",
    )
    parser.add_argument(
        "--long-run",
        type=Path,
        required=True,
        help="Veeksha output directory for the long-gap run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_output/experiment2_analysis"),
        help="Directory where the comparison artifacts will be written.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def resolve_run_dir(path: Path) -> Path:
    """Resolve either a timestamped Veeksha run dir or its parent output dir."""
    metrics_dir = path / "metrics"
    if metrics_dir.exists():
        return path

    candidates = sorted(
        [
            child
            for child in path.iterdir()
            if child.is_dir() and (child / "metrics").exists()
        ],
        key=lambda child: child.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        f"Could not find a Veeksha run directory under '{path}'. "
        "Pass either the timestamped run directory or its parent output dir."
    )


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize(values: Iterable[float]) -> dict[str, Any]:
    series = [float(value) for value in values]
    if not series:
        return {"count": 0}
    return {
        "count": len(series),
        "mean": statistics.fmean(series),
        "median": statistics.median(series),
        "p90": quantile(series, 0.90),
        "p95": quantile(series, 0.95),
        "p99": quantile(series, 0.99),
        "min": min(series),
        "max": max(series),
    }


def load_request_rows(run_dir: Path) -> list[dict[str, Any]]:
    resolved_run_dir = resolve_run_dir(run_dir)
    path = resolved_run_dir / "metrics" / "request_level_metrics.jsonl"
    rows = read_jsonl(path)
    rows.sort(
        key=lambda row: (
            int(row["session_id"]),
            float(row.get("scheduler_dispatched_at") or 0.0),
            int(row["request_id"]),
        )
    )
    return rows


def collect_probe_turn_rows(
    rows: list[dict[str, Any]],
    probe_session_ids: set[int],
) -> tuple[list[dict[str, Any]], tuple[float, float] | None]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["session_id"])].append(row)

    probe_turn_rows: list[dict[str, Any]] = []
    probe_dispatches: list[float] = []
    probe_completions: list[float] = []

    for session_id in sorted(probe_session_ids):
        session_rows = grouped.get(session_id, [])
        session_rows.sort(
            key=lambda row: (
                float(row.get("scheduler_dispatched_at") or 0.0),
                int(row["request_id"]),
            )
        )
        previous_completion: float | None = None
        for turn_idx, row in enumerate(session_rows):
            dispatched_at = maybe_float(row.get("scheduler_dispatched_at"))
            completed_at = maybe_float(row.get("client_completed_at"))
            realized_gap = None
            if previous_completion is not None and dispatched_at is not None:
                realized_gap = dispatched_at - previous_completion
            if completed_at is not None:
                previous_completion = completed_at

            enriched = dict(row)
            enriched["turn_idx"] = turn_idx
            enriched["realized_gap_s"] = realized_gap
            probe_turn_rows.append(enriched)

            if dispatched_at is not None:
                probe_dispatches.append(dispatched_at)
            if completed_at is not None:
                probe_completions.append(completed_at)

    if probe_dispatches and probe_completions:
        return probe_turn_rows, (min(probe_dispatches), max(probe_completions))
    return probe_turn_rows, None


def summarize_background_window(
    rows: list[dict[str, Any]],
    probe_session_ids: set[int],
    window: tuple[float, float] | None,
) -> dict[str, Any]:
    if window is None:
        return {"count": 0}
    start, end = window
    values = []
    for row in rows:
        session_id = int(row["session_id"])
        if session_id in probe_session_ids:
            continue
        dispatched_at = maybe_float(row.get("scheduler_dispatched_at"))
        ttfc = maybe_float(row.get("ttfc"))
        if dispatched_at is None or ttfc is None:
            continue
        if start <= dispatched_at <= end:
            values.append(ttfc)
    return summarize(values)


def per_turn_summary(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    buckets: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)  # type: ignore[return-value]
    )
    for row in rows:
        turn_idx = int(row["turn_idx"])
        ttfc = maybe_float(row.get("ttfc"))
        realized_gap = maybe_float(row.get("realized_gap_s"))
        total_prompt = maybe_float(row.get("num_total_prompt_tokens"))
        delta_prompt = maybe_float(row.get("target_num_delta_prompt_tokens"))

        if ttfc is not None:
            buckets[turn_idx]["ttfc"].append(ttfc)
        if realized_gap is not None:
            buckets[turn_idx]["realized_gap_s"].append(realized_gap)
        if total_prompt is not None:
            buckets[turn_idx]["num_total_prompt_tokens"].append(total_prompt)
        if delta_prompt is not None:
            buckets[turn_idx]["target_num_delta_prompt_tokens"].append(delta_prompt)

    summary: dict[int, dict[str, Any]] = {}
    for turn_idx, metrics in buckets.items():
        summary[turn_idx] = {
            metric_name: summarize(values) for metric_name, values in metrics.items()
        }
    return summary


def write_turn_comparison_csv(
    path: Path,
    *,
    short_summary: dict[int, dict[str, Any]],
    long_summary: dict[int, dict[str, Any]],
) -> None:
    fieldnames = [
        "turn_idx",
        "short_ttfc_median",
        "long_ttfc_median",
        "ttfc_median_delta",
        "short_gap_median",
        "long_gap_median",
        "short_total_prompt_median",
        "long_total_prompt_median",
        "short_delta_prompt_median",
        "long_delta_prompt_median",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for turn_idx in sorted(set(short_summary) | set(long_summary)):
            short_turn = short_summary.get(turn_idx, {})
            long_turn = long_summary.get(turn_idx, {})
            short_ttfc = short_turn.get("ttfc", {}).get("median")
            long_ttfc = long_turn.get("ttfc", {}).get("median")
            row = {
                "turn_idx": turn_idx,
                "short_ttfc_median": short_ttfc,
                "long_ttfc_median": long_ttfc,
                "ttfc_median_delta": (
                    None
                    if short_ttfc is None or long_ttfc is None
                    else long_ttfc - short_ttfc
                ),
                "short_gap_median": short_turn.get("realized_gap_s", {}).get("median"),
                "long_gap_median": long_turn.get("realized_gap_s", {}).get("median"),
                "short_total_prompt_median": short_turn.get(
                    "num_total_prompt_tokens", {}
                ).get("median"),
                "long_total_prompt_median": long_turn.get(
                    "num_total_prompt_tokens", {}
                ).get("median"),
                "short_delta_prompt_median": short_turn.get(
                    "target_num_delta_prompt_tokens", {}
                ).get("median"),
                "long_delta_prompt_median": long_turn.get(
                    "target_num_delta_prompt_tokens", {}
                ).get("median"),
            }
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(args.metadata)
    probe_session_ids = set(metadata["probe"]["expected_runtime_session_ids"])
    long_gap_turns = {
        int(turn_idx): float(gap_s)
        for turn_idx, gap_s in metadata["probe"]["long_gap_turns_s"].items()
    }

    short_rows = load_request_rows(args.short_run)
    long_rows = load_request_rows(args.long_run)

    short_probe_rows, short_probe_window = collect_probe_turn_rows(
        short_rows, probe_session_ids
    )
    long_probe_rows, long_probe_window = collect_probe_turn_rows(
        long_rows, probe_session_ids
    )

    short_turn_summary = per_turn_summary(short_probe_rows)
    long_turn_summary = per_turn_summary(long_probe_rows)

    comparison_summary = {
        "probe_sessions": sorted(probe_session_ids),
        "long_gap_turns_s": long_gap_turns,
        "short_probe_ttfc": summarize(
            row["ttfc"] for row in short_probe_rows if row.get("ttfc") is not None
        ),
        "long_probe_ttfc": summarize(
            row["ttfc"] for row in long_probe_rows if row.get("ttfc") is not None
        ),
        "short_background_ttfc_during_probe_window": summarize_background_window(
            short_rows, probe_session_ids, short_probe_window
        ),
        "long_background_ttfc_during_probe_window": summarize_background_window(
            long_rows, probe_session_ids, long_probe_window
        ),
        "per_gap_turn": {},
    }

    for turn_idx, gap_s in sorted(long_gap_turns.items()):
        comparison_summary["per_gap_turn"][turn_idx] = {
            "configured_gap_s": gap_s,
            "short_ttfc": short_turn_summary.get(turn_idx, {}).get("ttfc", {}),
            "long_ttfc": long_turn_summary.get(turn_idx, {}).get("ttfc", {}),
            "short_realized_gap_s": short_turn_summary.get(turn_idx, {}).get(
                "realized_gap_s", {}
            ),
            "long_realized_gap_s": long_turn_summary.get(turn_idx, {}).get(
                "realized_gap_s", {}
            ),
        }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    write_turn_comparison_csv(
        args.output_dir / "probe_turn_comparison.csv",
        short_summary=short_turn_summary,
        long_summary=long_turn_summary,
    )

    print("Wrote bursty-timing comparison artifacts to", args.output_dir)
    print("  summary:", args.output_dir / "summary.json")
    print("  turns:  ", args.output_dir / "probe_turn_comparison.csv")


if __name__ == "__main__":
    main()
