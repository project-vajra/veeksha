#!/usr/bin/env python3
"""Generate comparison artifacts for the workload-shape case study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from veeksha.case_studies.workload_shape_search import summarize_run


LINEAR_COLOR = "#1f4b99"
DAG_COLOR = "#b3472e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build blog-ready summary artifacts for the workload-shape case study "
            "from an existing rescored search directory."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing workload_shape_search_results.json. "
            "Typically the rescored search directory."
        ),
    )
    parser.add_argument(
        "--rate",
        type=float,
        help="Specific rate to analyze. Defaults to the best_rate in the results JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where comparison artifacts will be written.",
    )
    parser.add_argument(
        "--trace-metadata",
        type=Path,
        default=Path("traces/workload_shape/workload_shape_metadata.json"),
        help="Trace metadata file emitted by generate_workload_shape_traces.py.",
    )
    parser.add_argument(
        "--title",
        default="Workload Shape Comparison",
        help="Title prefix for the generated report.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}, got {type(row).__name__}."
                )
            rows.append(row)
    return rows


def resolve_run_dir(path: Path) -> Path:
    if (path / "metrics").exists():
        return path

    candidates = sorted(
        [child for child in path.iterdir() if child.is_dir() and (child / "metrics").exists()],
        key=lambda child: child.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        f"Could not resolve a Veeksha run directory from '{path}'."
    )


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def quantile(values: Iterable[float], q: float) -> Optional[float]:
    series = [float(value) for value in values]
    if not series:
        return None
    return float(np.quantile(np.asarray(series, dtype=float), q))


def metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        return np.asarray([]), np.asarray([])
    sorted_values = np.sort(np.asarray(values, dtype=float))
    cdf = np.arange(1, len(sorted_values) + 1, dtype=float) / len(sorted_values)
    return sorted_values, cdf


def percent_delta(lhs: Optional[float], rhs: Optional[float]) -> Optional[float]:
    if lhs is None or rhs is None or rhs == 0:
        return None
    return ((lhs / rhs) - 1.0) * 100.0


def fmt(value: Optional[float], spec: str) -> str:
    if value is None:
        return "n/a"
    return format(float(value), spec)


def load_selected_result(results_dir: Path, rate: Optional[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = read_json(results_dir / "workload_shape_search_results.json")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("Expected results list in workload_shape_search_results.json")

    selected_rate = float(rate) if rate is not None else payload.get("best_rate")
    if selected_rate is None:
        raise ValueError("No best_rate was recorded. Pass --rate explicitly.")

    for row in rows:
        if not isinstance(row, dict):
            continue
        row_rate = row.get("rate")
        if row_rate is None:
            continue
        if math.isclose(float(row_rate), float(selected_rate), rel_tol=0.0, abs_tol=1e-9):
            return payload, row

    raise ValueError(f"Rate {selected_rate} was not found in {results_dir}.")


def bar_plot(
    *,
    path: Path,
    title: str,
    labels: list[str],
    linear_values: list[Optional[float]],
    dag_values: list[Optional[float]],
    ylabel: str,
    value_formatter: str,
) -> None:
    indices = np.arange(len(labels), dtype=float)
    width = 0.36
    linear_series = [0.0 if value is None else float(value) for value in linear_values]
    dag_series = [0.0 if value is None else float(value) for value in dag_values]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(indices - width / 2, linear_series, width, label="Linear", color=LINEAR_COLOR)
    ax.bar(indices + width / 2, dag_series, width, label="DAG", color=DAG_COLOR)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for x, value in zip(indices - width / 2, linear_values, strict=True):
        if value is not None:
            ax.text(x, float(value), format(float(value), value_formatter), ha="center", va="bottom", fontsize=9)
    for x, value in zip(indices + width / 2, dag_values, strict=True):
        if value is not None:
            ax.text(x, float(value), format(float(value), value_formatter), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ecdf_plot(
    *,
    path: Path,
    title: str,
    linear_values: list[float],
    dag_values: list[float],
    xlabel: str,
) -> None:
    linear_x, linear_y = ecdf(linear_values)
    dag_x, dag_y = ecdf(dag_values)

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(linear_x):
        ax.step(linear_x, linear_y, where="post", color=LINEAR_COLOR, label="Linear", linewidth=2)
    if len(dag_x):
        ax.step(dag_x, dag_y, where="post", color=DAG_COLOR, label="DAG", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_markdown_report(
    *,
    path: Path,
    title: str,
    selected_rate: float,
    result_row: dict[str, Any],
    trace_metadata: Optional[dict[str, Any]],
    linear_summary: Any,
    dag_summary: Any,
    linear_rows: list[dict[str, Any]],
    dag_rows: list[dict[str, Any]],
    plots: list[Path],
) -> None:
    linear_total_prompt_p95 = quantile(metric_values(linear_rows, "num_total_prompt_tokens"), 0.95)
    dag_total_prompt_p95 = quantile(metric_values(dag_rows, "num_total_prompt_tokens"), 0.95)
    linear_delta_prompt_mean = linear_summary.mean_delta_prompt_tokens
    dag_delta_prompt_mean = dag_summary.mean_delta_prompt_tokens

    lines: list[str] = [f"# {title}", ""]
    lines.append(f"Selected shared session arrival rate: `{selected_rate:.2f}` sessions/s.")
    lines.append("")

    if trace_metadata:
        comparison = trace_metadata.get("comparison") or {}
        linear_workload = (trace_metadata.get("workloads") or {}).get("linear") or {}
        dag_workload = (trace_metadata.get("workloads") or {}).get("dag") or {}
        lines.append("## Matched fresh-token budget")
        lines.append("")
        lines.append(
            f"- Linear total new input tokens: `{linear_workload.get('total_new_input_tokens')}`"
        )
        lines.append(
            f"- DAG total new input tokens: `{dag_workload.get('total_new_input_tokens')}`"
        )
        lines.append(
            f"- Linear total output tokens: `{linear_workload.get('total_output_tokens')}`"
        )
        lines.append(
            f"- DAG total output tokens: `{dag_workload.get('total_output_tokens')}`"
        )
        lines.append(
            f"- DAG / linear effective-input ratio from the trace design: `{comparison.get('effective_input_ratio_dag_to_linear')}`"
        )
        lines.append(
            f"- DAG / linear cacheable-history ratio from the trace design: `{comparison.get('cacheable_history_ratio_dag_to_linear')}`"
        )
        lines.append("")

    lines.append("## Main finding")
    lines.append("")
    lines.append(
        "The workloads present the same fresh-token budget, but the DAG run forces the system "
        "to operate at longer effective context lengths and different cache behavior. "
        "That changes the reported performance even though user-visible work is matched."
    )
    lines.append("")
    lines.append("## Observed comparison")
    lines.append("")
    lines.append(
        f"- Mean fresh input tokens per request: linear `{fmt(linear_delta_prompt_mean, '.1f')}`, DAG `{fmt(dag_delta_prompt_mean, '.1f')}`"
    )
    lines.append(
        f"- Mean total prompt tokens per request: linear `{fmt(linear_summary.mean_total_prompt_tokens, '.1f')}`, DAG `{fmt(dag_summary.mean_total_prompt_tokens, '.1f')}`"
    )
    lines.append(
        f"- P95 total prompt tokens: linear `{fmt(linear_total_prompt_p95, '.1f')}`, DAG `{fmt(dag_total_prompt_p95, '.1f')}`"
    )
    lines.append(
        f"- TTFC p99: linear `{fmt(linear_summary.ttfc_p99_s, '.3f')}s`, DAG `{fmt(dag_summary.ttfc_p99_s, '.3f')}s`"
    )
    lines.append(
        f"- E2E p95: linear `{fmt(linear_summary.e2e_p95_s, '.3f')}s`, DAG `{fmt(dag_summary.e2e_p95_s, '.3f')}s`"
    )
    if linear_summary.decode_window_tbc_p99_s is not None and dag_summary.decode_window_tbc_p99_s is not None:
        lines.append(
            f"- Decode-window TBC p99: linear `{fmt(linear_summary.decode_window_tbc_p99_s * 1000.0, '.1f')} ms`, DAG `{fmt(dag_summary.decode_window_tbc_p99_s * 1000.0, '.1f')} ms`"
        )
    if linear_summary.vllm_prefix_cache_hit_rate is not None and dag_summary.vllm_prefix_cache_hit_rate is not None:
        lines.append(
            f"- vLLM prefix-cache hit rate: linear `{fmt(linear_summary.vllm_prefix_cache_hit_rate, '.3f')}`, DAG `{fmt(dag_summary.vllm_prefix_cache_hit_rate, '.3f')}`"
        )
    if linear_summary.vllm_prompt_cache_token_ratio is not None and dag_summary.vllm_prompt_cache_token_ratio is not None:
        lines.append(
            f"- vLLM prompt-cache token ratio: linear `{fmt(linear_summary.vllm_prompt_cache_token_ratio, '.3f')}`, DAG `{fmt(dag_summary.vllm_prompt_cache_token_ratio, '.3f')}`"
        )
    if linear_summary.vllm_kv_cache_usage_perc is not None and dag_summary.vllm_kv_cache_usage_perc is not None:
        lines.append(
            f"- vLLM KV-cache usage: linear `{fmt(linear_summary.vllm_kv_cache_usage_perc, '.3f')}`, DAG `{fmt(dag_summary.vllm_kv_cache_usage_perc, '.3f')}`"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"- DAG mean total prompt length is `{fmt(percent_delta(dag_summary.mean_total_prompt_tokens, linear_summary.mean_total_prompt_tokens), '.1f')}%` higher than linear."
    )
    lines.append(
        f"- DAG P95 total prompt length is `{fmt(percent_delta(dag_total_prompt_p95, linear_total_prompt_p95), '.1f')}%` higher than linear."
    )
    lines.append(
        f"- TTFC p99 shifts by `{fmt(percent_delta(dag_summary.ttfc_p99_s, linear_summary.ttfc_p99_s), '.1f')}%` between shapes at the same selected rate."
    )
    lines.append(
        f"- E2E p95 shifts by `{fmt(percent_delta(dag_summary.e2e_p95_s, linear_summary.e2e_p95_s), '.1f')}%` between shapes at the same selected rate."
    )
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    for plot_path in plots:
        lines.append(f"- `{plot_path.name}`")
    lines.append("")

    if result_row.get("notes"):
        lines.append("## Search notes")
        lines.append("")
        lines.append(f"- `{result_row['notes']}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or (results_dir / "analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, selected = load_selected_result(results_dir, args.rate)
    selected_rate = float(selected["rate"])

    linear_run_dir = resolve_run_dir(Path(str(selected["linear_run_dir"])).expanduser())
    dag_run_dir = resolve_run_dir(Path(str(selected["dag_run_dir"])).expanduser())

    linear_summary = summarize_run(
        workload="linear",
        rate=selected_rate,
        run_dir=str(linear_run_dir),
    )
    dag_summary = summarize_run(
        workload="dag",
        rate=selected_rate,
        run_dir=str(dag_run_dir),
    )

    linear_rows = read_jsonl(linear_run_dir / "metrics" / "request_level_metrics.jsonl")
    dag_rows = read_jsonl(dag_run_dir / "metrics" / "request_level_metrics.jsonl")

    trace_metadata = args.trace_metadata.resolve()
    trace_metadata_payload = read_json(trace_metadata) if trace_metadata.exists() else None

    latency_plot = output_dir / "latency_comparison.png"
    cache_plot = output_dir / "cache_comparison.png"
    fresh_prompt_plot = output_dir / "fresh_prompt_ecdf.png"
    total_prompt_plot = output_dir / "total_prompt_ecdf.png"

    bar_plot(
        path=latency_plot,
        title=f"{args.title}: latency at {selected_rate:.2f} sessions/s",
        labels=["TTFC p99 (s)", "E2E p95 (s)", "TBC p99 (ms)"],
        linear_values=[
            linear_summary.ttfc_p99_s,
            linear_summary.e2e_p95_s,
            (
                linear_summary.decode_window_tbc_p99_s * 1000.0
                if linear_summary.decode_window_tbc_p99_s is not None
                else None
            ),
        ],
        dag_values=[
            dag_summary.ttfc_p99_s,
            dag_summary.e2e_p95_s,
            (
                dag_summary.decode_window_tbc_p99_s * 1000.0
                if dag_summary.decode_window_tbc_p99_s is not None
                else None
            ),
        ],
        ylabel="Latency",
        value_formatter=".2f",
    )

    bar_plot(
        path=cache_plot,
        title=f"{args.title}: cache and context at {selected_rate:.2f} sessions/s",
        labels=["Total prompt mean", "Cacheable mean", "Prefix hit", "Prompt cache", "KV usage"],
        linear_values=[
            linear_summary.mean_total_prompt_tokens,
            linear_summary.mean_cacheable_prompt_tokens,
            linear_summary.vllm_prefix_cache_hit_rate,
            linear_summary.vllm_prompt_cache_token_ratio,
            linear_summary.vllm_kv_cache_usage_perc,
        ],
        dag_values=[
            dag_summary.mean_total_prompt_tokens,
            dag_summary.mean_cacheable_prompt_tokens,
            dag_summary.vllm_prefix_cache_hit_rate,
            dag_summary.vllm_prompt_cache_token_ratio,
            dag_summary.vllm_kv_cache_usage_perc,
        ],
        ylabel="Value",
        value_formatter=".2f",
    )

    ecdf_plot(
        path=fresh_prompt_plot,
        title=f"{args.title}: fresh prompt tokens per request",
        linear_values=metric_values(linear_rows, "num_delta_prompt_tokens"),
        dag_values=metric_values(dag_rows, "num_delta_prompt_tokens"),
        xlabel="Fresh prompt tokens",
    )
    ecdf_plot(
        path=total_prompt_plot,
        title=f"{args.title}: effective prompt tokens per request",
        linear_values=metric_values(linear_rows, "num_total_prompt_tokens"),
        dag_values=metric_values(dag_rows, "num_total_prompt_tokens"),
        xlabel="Total prompt tokens",
    )

    summary_payload = {
        "rate": selected_rate,
        "result_status": selected.get("status"),
        "result_healthy": selected.get("healthy"),
        "linear_run_dir": str(linear_run_dir),
        "dag_run_dir": str(dag_run_dir),
        "linear": {
            "ttfc_p99_s": linear_summary.ttfc_p99_s,
            "e2e_p95_s": linear_summary.e2e_p95_s,
            "tbc_p99_ms": (
                linear_summary.decode_window_tbc_p99_s * 1000.0
                if linear_summary.decode_window_tbc_p99_s is not None
                else None
            ),
            "mean_delta_prompt_tokens": linear_summary.mean_delta_prompt_tokens,
            "mean_total_prompt_tokens": linear_summary.mean_total_prompt_tokens,
            "mean_cacheable_prompt_tokens": linear_summary.mean_cacheable_prompt_tokens,
            "vllm_prefix_cache_hit_rate": linear_summary.vllm_prefix_cache_hit_rate,
            "vllm_prompt_cache_token_ratio": linear_summary.vllm_prompt_cache_token_ratio,
            "vllm_kv_cache_usage_perc": linear_summary.vllm_kv_cache_usage_perc,
        },
        "dag": {
            "ttfc_p99_s": dag_summary.ttfc_p99_s,
            "e2e_p95_s": dag_summary.e2e_p95_s,
            "tbc_p99_ms": (
                dag_summary.decode_window_tbc_p99_s * 1000.0
                if dag_summary.decode_window_tbc_p99_s is not None
                else None
            ),
            "mean_delta_prompt_tokens": dag_summary.mean_delta_prompt_tokens,
            "mean_total_prompt_tokens": dag_summary.mean_total_prompt_tokens,
            "mean_cacheable_prompt_tokens": dag_summary.mean_cacheable_prompt_tokens,
            "vllm_prefix_cache_hit_rate": dag_summary.vllm_prefix_cache_hit_rate,
            "vllm_prompt_cache_token_ratio": dag_summary.vllm_prompt_cache_token_ratio,
            "vllm_kv_cache_usage_perc": dag_summary.vllm_kv_cache_usage_perc,
        },
        "relative_deltas": {
            "total_prompt_mean_pct": percent_delta(
                dag_summary.mean_total_prompt_tokens,
                linear_summary.mean_total_prompt_tokens,
            ),
            "cacheable_prompt_mean_pct": percent_delta(
                dag_summary.mean_cacheable_prompt_tokens,
                linear_summary.mean_cacheable_prompt_tokens,
            ),
            "ttfc_p99_pct": percent_delta(
                dag_summary.ttfc_p99_s,
                linear_summary.ttfc_p99_s,
            ),
            "e2e_p95_pct": percent_delta(
                dag_summary.e2e_p95_s,
                linear_summary.e2e_p95_s,
            ),
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = output_dir / "blog_summary.md"
    build_markdown_report(
        path=report_path,
        title=args.title,
        selected_rate=selected_rate,
        result_row=selected,
        trace_metadata=trace_metadata_payload,
        linear_summary=linear_summary,
        dag_summary=dag_summary,
        linear_rows=linear_rows,
        dag_rows=dag_rows,
        plots=[latency_plot, cache_plot, fresh_prompt_plot, total_prompt_plot],
    )

    print("Wrote workload-shape comparison artifacts to", output_dir)
    print("  summary:", summary_path)
    print("  report: ", report_path)
    print("  plots:  ", latency_plot, cache_plot, fresh_prompt_plot, total_prompt_plot)


if __name__ == "__main__":
    main()
