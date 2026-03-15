#!/usr/bin/env python3
"""Generate blog-ready artifacts for the deployment-claim case study."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Optional


def _is_veeksha_repo(path: Path) -> bool:
    return (path / "veeksha" / "case_studies").exists() and (path / "pyproject.toml").exists()


def _find_veeksha_repo() -> Path | None:
    script_path = Path(__file__).resolve()
    candidates: list[Path] = []

    env_repo = os.environ.get("VEEKSHA_REPO")
    if env_repo:
        candidates.append(Path(env_repo).expanduser().resolve())

    for parent in script_path.parents:
        candidates.append(parent)
        candidates.append(parent / "veeksha")
        candidates.append(parent / "veeksha-prs")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_veeksha_repo(candidate):
            return candidate
    return None


def _bootstrap_veeksha_python() -> None:
    veeksha_repo = _find_veeksha_repo()
    if veeksha_repo is None:
        return

    candidate = veeksha_repo / ".venv" / "bin" / "python"
    if candidate.exists() and Path(sys.executable).resolve() != candidate.resolve():
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/tmp")
        os.execve(
            str(candidate),
            [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )


_bootstrap_veeksha_python()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_JSON_NAME = "workload_deployment_claim_results.json"
LINEAR_COLOR = "#1f4b99"
DAG_COLOR = "#b3472e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build summary artifacts for the rate-normalized deployment-claim "
            "case study from completed linear and DAG benchmark runs."
        )
    )
    parser.add_argument("--linear-search-dir", type=Path, required=True)
    parser.add_argument("--dag-replay-dir", type=Path, required=True)
    parser.add_argument("--dag-search-dir", type=Path, required=True)
    parser.add_argument(
        "--trace-metadata",
        type=Path,
        default=Path("traces/workload_deployment_claim_case_study/workload_metadata.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_output/workload_deployment_claim_case_study/analysis"),
    )
    parser.add_argument(
        "--title",
        default="Deployment Claim Case Study",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def percent_delta(lhs: Optional[float], rhs: Optional[float]) -> Optional[float]:
    if lhs is None or rhs is None or rhs == 0:
        return None
    return ((lhs / rhs) - 1.0) * 100.0


def fmt(value: Optional[float], spec: str) -> str:
    if value is None:
        return "n/a"
    return format(float(value), spec)


def load_result_payload(results_dir: Path) -> dict[str, Any]:
    return read_json(results_dir / RESULTS_JSON_NAME)


def extract_best_result(payload: dict[str, Any]) -> dict[str, Any]:
    best = payload.get("best_result")
    if isinstance(best, dict):
        return best

    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty results list.")

    healthy = [row for row in rows if isinstance(row, dict) and row.get("healthy")]
    if healthy:
        return max(
            healthy,
            key=lambda row: float(row.get("normalized_request_rate", 0.0)),
        )

    row = rows[-1]
    if not isinstance(row, dict):
        raise ValueError("Expected the last result row to be an object.")
    return row


def extract_single_result(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("Expected a single-result payload for the replay run.")
    return rows[0]


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
    ax.bar(indices - width / 2, linear_series, width, label="Linear tuned", color=LINEAR_COLOR)
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dollar_sentence(capacity_uplift: Optional[float], gpu_hour_price_usd: Optional[float]) -> str:
    if capacity_uplift is None:
        return (
            "If one reference deployment costs `$X` per hour, this gap implies about "
            "`$X * (capacity_uplift - 1)` extra per hour to hold the same SLO under DAG traffic."
        )
    if gpu_hour_price_usd is None:
        return (
            "If one reference deployment costs `$X` per hour, this gap implies about "
            f"`$X * ({capacity_uplift:.3f} - 1)` extra per hour to hold the same SLO under DAG traffic."
        )
    extra = gpu_hour_price_usd * max(capacity_uplift - 1.0, 0.0)
    return (
        f"If one reference deployment costs `${gpu_hour_price_usd:.2f}` per hour, this gap implies about "
        f"`${extra:.2f}` extra per hour to hold the same SLO under DAG traffic."
    )


def build_markdown_report(
    *,
    path: Path,
    title: str,
    linear_best: dict[str, Any],
    dag_replay: dict[str, Any],
    dag_best: dict[str, Any],
    summary: dict[str, Any],
    gpu_hour_price_usd: Optional[float],
) -> None:
    replay_notes = dag_replay.get("notes") or []
    replay_note_text = ", ".join(str(note) for note in replay_notes) if replay_notes else "none"
    capacity_uplift = maybe_float(summary["capacity_uplift"].get("normalized_request_rate"))

    lines: list[str] = [f"# {title}", ""]
    lines.append("## Deployment setup")
    lines.append("")
    lines.append(
        f"- Linear tuned normalized request rate (`rho_A*`): `{fmt(maybe_float(linear_best.get('normalized_request_rate')), '.2f')}` req/s"
    )
    lines.append(
        f"- DAG replay normalized request rate: `{fmt(maybe_float(dag_replay.get('normalized_request_rate')), '.2f')}` req/s"
    )
    lines.append(
        f"- DAG max healthy normalized request rate (`rho_B*`): `{fmt(maybe_float(dag_best.get('normalized_request_rate')), '.2f')}` req/s"
    )
    lines.append("")

    lines.append("## Replay outcome at the linear-tuned budget")
    lines.append("")
    lines.append(
        f"- DAG replay healthy: `{bool(dag_replay.get('healthy'))}`"
    )
    lines.append(
        f"- Replay guardrail notes: `{replay_note_text}`"
    )
    lines.append(
        f"- TTFC p99: linear `{fmt(maybe_float(linear_best.get('run_ttfc_p99_s')), '.3f')}s`, DAG replay `{fmt(maybe_float(dag_replay.get('run_ttfc_p99_s')), '.3f')}s`"
    )
    lines.append(
        f"- E2E p95: linear `{fmt(maybe_float(linear_best.get('run_e2e_p95_s')), '.3f')}s`, DAG replay `{fmt(maybe_float(dag_replay.get('run_e2e_p95_s')), '.3f')}s`"
    )
    lines.append(
        f"- Decode-window TBC p99: linear `{fmt(maybe_float(linear_best.get('run_decode_window_tbc_p99_s')) * 1000.0 if linear_best.get('run_decode_window_tbc_p99_s') is not None else None, '.1f')} ms`, DAG replay `{fmt(maybe_float(dag_replay.get('run_decode_window_tbc_p99_s')) * 1000.0 if dag_replay.get('run_decode_window_tbc_p99_s') is not None else None, '.1f')} ms`"
    )
    lines.append("")

    lines.append("## Capacity implication")
    lines.append("")
    lines.append(
        f"- Capacity uplift (`rho_A* / rho_B*`): `{fmt(capacity_uplift, '.3f')}x`"
    )
    lines.append(
        f"- Fresh-input-rate uplift: `{fmt(maybe_float(summary['capacity_uplift'].get('fresh_input_tokens_per_s')), '.3f')}x`"
    )
    lines.append(
        f"- Requested-output-rate uplift: `{fmt(maybe_float(summary['capacity_uplift'].get('requested_output_tokens_per_s')), '.3f')}x`"
    )
    lines.append("")

    lines.append("## Dollar framing")
    lines.append("")
    lines.append(dollar_sentence(capacity_uplift, gpu_hour_price_usd))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    linear_payload = load_result_payload(args.linear_search_dir.resolve())
    dag_replay_payload = load_result_payload(args.dag_replay_dir.resolve())
    dag_search_payload = load_result_payload(args.dag_search_dir.resolve())
    trace_metadata = read_json(args.trace_metadata.resolve())

    linear_best = extract_best_result(linear_payload)
    dag_replay = extract_single_result(dag_replay_payload)
    dag_best = extract_best_result(dag_search_payload)

    if not linear_best.get("healthy"):
        raise ValueError("The selected linear frontier run is not healthy.")
    if not dag_best.get("healthy"):
        raise ValueError("The selected DAG frontier run is not healthy.")

    rho_a = maybe_float(linear_best.get("normalized_request_rate"))
    rho_replay = maybe_float(dag_replay.get("normalized_request_rate"))
    rho_b = maybe_float(dag_best.get("normalized_request_rate"))
    if rho_a is None or rho_replay is None or rho_b is None:
        raise ValueError("Missing normalized request rate in one or more result rows.")

    fresh_input_per_request = maybe_float(
        ((trace_metadata.get("rate_model") or {}).get("fresh_input_tokens_per_request"))
    )
    requested_output_per_request = maybe_float(
        ((trace_metadata.get("rate_model") or {}).get("output_tokens_per_request"))
    )
    if fresh_input_per_request is None or requested_output_per_request is None:
        raise ValueError("Trace metadata is missing the rate model token counts.")

    same_budget = math.isclose(rho_a, rho_replay, rel_tol=0.0, abs_tol=1e-9)
    if not same_budget:
        raise ValueError(
            "The DAG replay rate does not match the linear tuned rate. "
            f"Expected {rho_a}, got {rho_replay}."
        )

    capacity_uplift = rho_a / rho_b if rho_b > 0 else None
    fresh_input_uplift = (
        (fresh_input_per_request * rho_a) / (fresh_input_per_request * rho_b)
        if rho_b > 0
        else None
    )
    requested_output_uplift = (
        (requested_output_per_request * rho_a)
        / (requested_output_per_request * rho_b)
        if rho_b > 0
        else None
    )

    normalized_rate_table = {
        "linear_tuned": {
            "normalized_request_rate": rho_a,
            "derived_session_rate": maybe_float(linear_best.get("derived_session_rate")),
            "fresh_input_tokens_per_s": maybe_float(linear_best.get("fresh_input_tokens_per_s")),
            "requested_output_tokens_per_s": maybe_float(linear_best.get("requested_output_tokens_per_s")),
        },
        "dag_replay_at_linear_budget": {
            "normalized_request_rate": rho_replay,
            "derived_session_rate": maybe_float(dag_replay.get("derived_session_rate")),
            "fresh_input_tokens_per_s": maybe_float(dag_replay.get("fresh_input_tokens_per_s")),
            "requested_output_tokens_per_s": maybe_float(dag_replay.get("requested_output_tokens_per_s")),
        },
        "dag_max_healthy": {
            "normalized_request_rate": rho_b,
            "derived_session_rate": maybe_float(dag_best.get("derived_session_rate")),
            "fresh_input_tokens_per_s": maybe_float(dag_best.get("fresh_input_tokens_per_s")),
            "requested_output_tokens_per_s": maybe_float(dag_best.get("requested_output_tokens_per_s")),
        },
    }

    summary = {
        "title": args.title,
        "trace_metadata_path": str(args.trace_metadata.resolve()),
        "linear_search_dir": str(args.linear_search_dir.resolve()),
        "dag_replay_dir": str(args.dag_replay_dir.resolve()),
        "dag_search_dir": str(args.dag_search_dir.resolve()),
        "same_useful_work_rate": same_budget,
        "linear_tuned": linear_best,
        "dag_replay_at_linear_budget": dag_replay,
        "dag_max_healthy": dag_best,
        "capacity_uplift": {
            "normalized_request_rate": capacity_uplift,
            "fresh_input_tokens_per_s": fresh_input_uplift,
            "requested_output_tokens_per_s": requested_output_uplift,
        },
        "replay_relative_deltas": {
            "ttfc_p99_pct": percent_delta(
                maybe_float(dag_replay.get("run_ttfc_p99_s")),
                maybe_float(linear_best.get("run_ttfc_p99_s")),
            ),
            "e2e_p95_pct": percent_delta(
                maybe_float(dag_replay.get("run_e2e_p95_s")),
                maybe_float(linear_best.get("run_e2e_p95_s")),
            ),
            "tbc_p99_pct": percent_delta(
                maybe_float(dag_replay.get("run_decode_window_tbc_p99_s")),
                maybe_float(linear_best.get("run_decode_window_tbc_p99_s")),
            ),
            "tpot_based_throughput_pct": percent_delta(
                maybe_float(dag_replay.get("run_tpot_based_throughput")),
                maybe_float(linear_best.get("run_tpot_based_throughput")),
            ),
        },
        "gpu_hour_price_usd": maybe_float(
            linear_payload.get("gpu_hour_price_usd") or dag_search_payload.get("gpu_hour_price_usd")
        ),
    }

    slo_plot = output_dir / "slo_comparison_at_linear_budget.png"
    capacity_plot = output_dir / "capacity_uplift.png"

    bar_plot(
        path=slo_plot,
        title=f"{args.title}: SLO metrics at rho_A*",
        labels=["TTFC p99 (s)", "E2E p95 (s)", "TBC p99 (ms)"],
        linear_values=[
            maybe_float(linear_best.get("run_ttfc_p99_s")),
            maybe_float(linear_best.get("run_e2e_p95_s")),
            (
                maybe_float(linear_best.get("run_decode_window_tbc_p99_s")) * 1000.0
                if linear_best.get("run_decode_window_tbc_p99_s") is not None
                else None
            ),
        ],
        dag_values=[
            maybe_float(dag_replay.get("run_ttfc_p99_s")),
            maybe_float(dag_replay.get("run_e2e_p95_s")),
            (
                maybe_float(dag_replay.get("run_decode_window_tbc_p99_s")) * 1000.0
                if dag_replay.get("run_decode_window_tbc_p99_s") is not None
                else None
            ),
        ],
        ylabel="Latency",
        value_formatter=".2f",
    )

    bar_plot(
        path=capacity_plot,
        title=f"{args.title}: capacity frontier ({fmt(capacity_uplift, '.2f')}x uplift)",
        labels=["Request rate", "Fresh input tok/s", "Requested output tok/s"],
        linear_values=[
            rho_a,
            maybe_float(linear_best.get("fresh_input_tokens_per_s")),
            maybe_float(linear_best.get("requested_output_tokens_per_s")),
        ],
        dag_values=[
            rho_b,
            maybe_float(dag_best.get("fresh_input_tokens_per_s")),
            maybe_float(dag_best.get("requested_output_tokens_per_s")),
        ],
        ylabel="Rate",
        value_formatter=".2f",
    )

    write_json(output_dir / "normalized_rate_table.json", normalized_rate_table)
    write_json(output_dir / "summary.json", summary)
    build_markdown_report(
        path=output_dir / "blog_summary.md",
        title=args.title,
        linear_best=linear_best,
        dag_replay=dag_replay,
        dag_best=dag_best,
        summary=summary,
        gpu_hour_price_usd=maybe_float(summary.get("gpu_hour_price_usd")),
    )


if __name__ == "__main__":
    main()
