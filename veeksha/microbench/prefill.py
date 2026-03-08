"""Prefill microbenchmark: build, validate, and report."""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.table import Table

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.config.generator.channel import TextChannelGeneratorConfig
from veeksha.config.generator.length import (
    FixedLengthGeneratorConfig,
    StairLengthGeneratorConfig,
)
from veeksha.config.generator.requested_output import (
    OutputSpecConfig,
    TextOutputSpecConfig,
)
from veeksha.config.generator.session import SyntheticSessionGeneratorConfig
from veeksha.config.generator.session_graph import (
    SingleRequestSessionGraphGeneratorConfig,
)
from veeksha.config.runtime import RuntimeConfig
from veeksha.config.trace_recorder import TraceRecorderConfig
from veeksha.config.traffic import ConcurrentTrafficConfig
from veeksha.microbench.common import (
    ValidationResult,
    build_client_config,
    compute_stats,
    console,
    find_all_run_metrics,
    fmt_ms,
    load_request_metrics,
    save_json,
)
from veeksha.microbench.config import PrefillMicrobenchmarkConfig

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER_ROWS: list[tuple[str, str]] = [
    ("Input lengths", "input_lengths"),
    ("Output tokens", "output_tokens"),
    ("Samples/length", "samples_per_length"),
]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_benchmark_configs(cfg: PrefillMicrobenchmarkConfig) -> list[BenchmarkConfig]:
    total_sessions = len(cfg.input_lengths) * cfg.samples_per_length
    return [
        BenchmarkConfig(
            output_dir=cfg.output_dir,
            seed=cfg.seed,
            session_generator=SyntheticSessionGeneratorConfig(
                session_graph=SingleRequestSessionGraphGeneratorConfig(),
                channels=[
                    TextChannelGeneratorConfig(
                        body_length_generator=StairLengthGeneratorConfig(
                            values=cfg.input_lengths,
                            repeat_each=cfg.samples_per_length,
                            wrap=False,
                        ),
                        shared_prefix_ratio=0.0,
                    )
                ],
                output_spec=OutputSpecConfig(
                    text=TextOutputSpecConfig(
                        output_length_generator=FixedLengthGeneratorConfig(
                            value=cfg.output_tokens,
                        ),
                    ),
                ),
            ),
            traffic_scheduler=ConcurrentTrafficConfig(
                target_concurrent_sessions=1,
                rampup_seconds=0,
                cancel_session_on_failure=False,
            ),
            evaluators=[
                PerformanceEvaluatorConfig(stream_metrics=False, slos=[]),
            ],
            client=build_client_config(cfg),
            runtime=RuntimeConfig(
                max_sessions=total_sessions,
                benchmark_timeout=cfg.benchmark_timeout,
                pregenerate_sessions=True,
            ),
            trace_recorder=TraceRecorderConfig(enabled=False),
        )
    ]


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate(cfg: PrefillMicrobenchmarkConfig, output_dir: str) -> ValidationResult:
    result = ValidationResult()
    metrics = load_request_metrics(output_dir)
    if metrics is None:
        result.fail("metrics_found", "No request_level_metrics.jsonl found")
        return result
    result.passed("metrics_found")

    expected_count = len(cfg.input_lengths) * cfg.samples_per_length

    if len(metrics) == expected_count:
        result.passed("session_count", f"{len(metrics)} requests")
    else:
        result.fail("session_count", f"expected {expected_count}, got {len(metrics)}")

    failed_requests = [r for r in metrics if r.get("num_output_tokens", 0) == 0]
    if not failed_requests:
        result.passed("no_errors", "all requests produced output")
    else:
        result.fail(
            "no_errors", f"{len(failed_requests)} requests produced 0 output tokens"
        )

    mismatched = [r for r in metrics if r["num_output_tokens"] != cfg.output_tokens]
    if not mismatched:
        result.passed(
            "output_tokens", f"all requests produced {cfg.output_tokens} output tokens"
        )
    else:
        result.warn(
            "output_tokens",
            f"{len(mismatched)} requests had unexpected output token count",
        )

    sorted_by_session = sorted(metrics, key=lambda r: r["session_id"])
    is_sequential = True
    for i in range(1, len(sorted_by_session)):
        prev_done = sorted_by_session[i - 1]["client_completed_at"]
        curr_start = sorted_by_session[i]["scheduler_dispatched_at"]
        if curr_start < prev_done - 0.01:
            is_sequential = False
            break
    if is_sequential:
        result.passed("sequential_execution", "requests executed one at a time")
    else:
        result.warn(
            "sequential_execution", "some requests overlapped (concurrent != 1?)"
        )

    for i, record in enumerate(sorted_by_session):
        length_idx = i // cfg.samples_per_length
        if length_idx < len(cfg.input_lengths):
            expected_prompt = cfg.input_lengths[length_idx]
            actual_prompt = record["target_num_delta_prompt_tokens"]
            if actual_prompt != expected_prompt:
                result.warn(
                    "prompt_tokens_stair",
                    f"session {record['session_id']}: expected {expected_prompt}, got {actual_prompt}",
                )
                break
    else:
        result.passed(
            "prompt_tokens_stair", "prompt tokens follow expected stair pattern"
        )

    return result


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def print_results_table(cfg: PrefillMicrobenchmarkConfig) -> None:
    all_runs = find_all_run_metrics(cfg.output_dir)
    if not all_runs:
        return

    metrics = [record for _, records in all_runs for record in records]

    ttfc_by_length: dict[int, list[float]] = {}
    for record in metrics:
        il = record["target_num_delta_prompt_tokens"]
        ttfc = record.get("ttfc")
        if ttfc is not None:
            ttfc_by_length.setdefault(il, []).append(ttfc)

    if not ttfc_by_length:
        return

    rows = [
        {"input_length": il, "ttfc": compute_stats(vals)}
        for il, vals in sorted(ttfc_by_length.items())
    ]

    table = Table(title="Prefill Results (TTFC)")
    table.add_column("Input Length", justify="right", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")
    table.add_column("Min (ms)", justify="right", style="dim")
    table.add_column("Max (ms)", justify="right", style="dim")
    table.add_column("Count", justify="right", style="dim")

    for row in rows:
        s = row["ttfc"]
        table.add_row(
            str(row["input_length"]),
            fmt_ms(s["mean"]),
            fmt_ms(s["median"]),
            fmt_ms(s["p99"]),
            fmt_ms(s["min"]),
            fmt_ms(s["max"]),
            str(s["count"]),
        )

    console.print()
    console.print(table)
    console.print()

    save_json(
        {"type": "prefill", "metric": "ttfc", "results": rows},
        os.path.join(cfg.output_dir, "prefill_results.json"),
    )

    # CSV
    csv_path = os.path.join(cfg.output_dir, "prefill_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_length",
                "mean_ms",
                "p50_ms",
                "p99_ms",
                "min_ms",
                "max_ms",
                "count",
            ],
        )
        writer.writeheader()
        for row in rows:
            s = row["ttfc"]
            writer.writerow(
                {
                    "input_length": row["input_length"],
                    "mean_ms": s["mean"] * 1000,
                    "p50_ms": s["median"] * 1000,
                    "p99_ms": s["p99"] * 1000,
                    "min_ms": s["min"] * 1000,
                    "max_ms": s["max"] * 1000,
                    "count": s["count"],
                }
            )
    console.print(f"  CSV saved to {csv_path}")

    # Plots
    if len(rows) >= 2:
        plots_dir = os.path.join(cfg.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        lengths = [r["input_length"] for r in rows]
        p50s = [r["ttfc"]["median"] * 1000 for r in rows]
        p99s = [r["ttfc"]["p99"] * 1000 for r in rows]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lengths, p50s, "o-", label="P50", linewidth=2)
        ax.plot(lengths, p99s, "s--", label="P99", linewidth=2)
        ax.set_xlabel("Input Length (tokens)")
        ax.set_ylabel("TTFC (ms)")
        ax.set_title("Time to First Token vs Input Length")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "ttfc_vs_input_length.png"), dpi=150)
        plt.close(fig)
        console.print(f"  Plots saved to {plots_dir}/")

    console.print()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def run_prefill(cfg: PrefillMicrobenchmarkConfig) -> None:
    """Run a single prefill microbenchmark."""
    from veeksha.microbench.runner import run

    run(cfg, "prefill", BANNER_ROWS, build_benchmark_configs, print_results_table, validate)


def main() -> None:
    for cfg in PrefillMicrobenchmarkConfig.create_from_cli_args():
        run_prefill(cfg)
