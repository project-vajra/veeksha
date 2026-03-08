"""Decode microbenchmark: build, validate, and report."""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.table import Table

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.evaluator import (
    DecodeWindowConfig,
    PerformanceEvaluatorConfig,
    TextChannelPerformanceConfig,
)
from veeksha.config.generator.channel import TextChannelGeneratorConfig
from veeksha.config.generator.length import FixedLengthGeneratorConfig
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
from veeksha.config.traffic import SequentialLaunchTrafficConfig
from veeksha.microbench.common import (
    _OUTPUT_TOKEN_MULTIPLIER,
    ValidationResult,
    build_client_config,
    compute_prefill_iterations,
    compute_stats,
    console,
    find_all_run_metrics,
    fmt_ms,
    load_decode_window_json,
    load_decode_window_stats,
    load_request_metrics,
    save_json,
)
from veeksha.microbench.config import DecodeMicrobenchmarkConfig

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER_ROWS: list[tuple[str, str]] = [
    ("Batch sizes", "batch_sizes"),
    ("Input lengths", "input_lengths"),
    ("Samples/length", "samples_per_length"),
    ("Chunk size", "engine_chunk_size"),
]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def required_decode_output_tokens(
    samples_per_length: int,
    batch_size: int,
    input_length: int,
    chunk_size: int,
) -> int:
    """Compute output tokens for a decode benchmark run.

    Request 0 enters decode first and must still be generating when the
    last request finishes prefilling, plus *samples_per_length* additional
    pure-decode iterations for measurement.

        output_tokens = samples_per_length
            + (batch_size - 1) * ceil(input_length / (chunk_size - batch_size))
    """
    if batch_size == 1:
        return samples_per_length
    ramp_up = (batch_size - 1) * compute_prefill_iterations(
        input_length, chunk_size, batch_size
    )
    return samples_per_length + ramp_up


def build_benchmark_configs(cfg: DecodeMicrobenchmarkConfig) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    for batch_size in cfg.batch_sizes:
        for input_length in cfg.input_lengths:
            output_tokens = (
                required_decode_output_tokens(
                    cfg.samples_per_length,
                    batch_size,
                    input_length,
                    cfg.engine_chunk_size,
                )
                * _OUTPUT_TOKEN_MULTIPLIER
            )
            configs.append(
                BenchmarkConfig(
                    output_dir=f"{cfg.output_dir}/bs={batch_size}_il={input_length}",
                    seed=cfg.seed,
                    session_generator=SyntheticSessionGeneratorConfig(
                        session_graph=SingleRequestSessionGraphGeneratorConfig(),
                        channels=[
                            TextChannelGeneratorConfig(
                                body_length_generator=FixedLengthGeneratorConfig(
                                    value=input_length,
                                ),
                            )
                        ],
                        output_spec=OutputSpecConfig(
                            text=TextOutputSpecConfig(
                                output_length_generator=FixedLengthGeneratorConfig(
                                    value=output_tokens,
                                ),
                            ),
                        ),
                    ),
                    traffic_scheduler=SequentialLaunchTrafficConfig(
                        cancel_session_on_failure=False,
                    ),
                    evaluators=[
                        PerformanceEvaluatorConfig(
                            stream_metrics=False,
                            slos=[],
                            text_channel=TextChannelPerformanceConfig(
                                decode_window_enabled=True,
                                decode_window_config=DecodeWindowConfig(
                                    min_active_requests="max_observed",
                                    selection_strategy="all",
                                ),
                            ),
                        ),
                    ],
                    client=build_client_config(cfg),
                    runtime=RuntimeConfig(
                        max_sessions=batch_size,
                        num_client_threads=batch_size,
                        benchmark_timeout=cfg.benchmark_timeout,
                        pregenerate_sessions=True,
                    ),
                    trace_recorder=TraceRecorderConfig(enabled=False),
                )
            )
    return configs


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate(cfg: DecodeMicrobenchmarkConfig, output_dir: str) -> ValidationResult:
    result = ValidationResult()
    for batch_size in cfg.batch_sizes:
        for input_length in cfg.input_lengths:
            label = f"bs={batch_size},il={input_length}"
            _validate_one_run(result, cfg, batch_size, input_length, output_dir, label)
    return result


def _validate_one_run(
    result: ValidationResult,
    cfg: DecodeMicrobenchmarkConfig,
    batch_size: int,
    input_length: int,
    output_dir: str,
    label: str,
) -> None:
    metrics = load_request_metrics(f"{output_dir}/bs={batch_size}_il={input_length}")
    if metrics is None:
        result.fail(
            f"metrics_found [{label}]",
            "No request_level_metrics.jsonl found in decode dir",
        )
        return

    matching = [
        r for r in metrics if r["target_num_delta_prompt_tokens"] == input_length
    ]
    if not matching:
        result.warn(
            f"matching_requests [{label}]", f"no requests with prompt={input_length}"
        )
        return
    result.passed(f"matching_requests [{label}]", f"{len(matching)} requests")

    sorted_by_session = sorted(matching, key=lambda r: r["session_id"])
    first_token_times = [
        r["client_picked_up_at"] + r["ttfc"] for r in sorted_by_session
    ]
    out_of_order = sum(
        1
        for i in range(1, len(first_token_times))
        if first_token_times[i] < first_token_times[i - 1] - 0.005
    )
    if out_of_order == 0:
        result.passed(f"fcfs_order [{label}]", "first tokens arrived in session order")
    else:
        result.warn(
            f"fcfs_order [{label}]",
            f"{out_of_order} out-of-order first tokens (engine may batch prefills)",
        )

    dw_data = load_decode_window_json(f"{output_dir}/bs={batch_size}_il={input_length}")
    if dw_data is not None:
        num_segments = dw_data.get("windows", {}).get("num_selected_segments", 0)
        tbc_count = dw_data.get("tbc_in_window_stats", {}).get("count", 0)
        if num_segments == 0 or tbc_count == 0:
            result.fail(
                f"decode_window_overlap [{label}]",
                "no qualifying decode windows found — increase output tokens",
            )
        elif tbc_count < cfg.samples_per_length:
            result.warn(
                f"decode_window_overlap [{label}]",
                f"low sample count in decode window: {tbc_count} < {cfg.samples_per_length}",
            )
        else:
            result.passed(
                f"decode_window_overlap [{label}]",
                f"{tbc_count} samples in decode window",
            )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def print_results_table(cfg: DecodeMicrobenchmarkConfig) -> None:
    all_runs = find_all_run_metrics(cfg.output_dir)
    if not all_runs:
        return

    rows: list[tuple[int, int, dict]] = []
    for metrics_dir, records in all_runs:
        batch_size = len(records)
        input_length = records[0]["target_num_delta_prompt_tokens"]

        dw_stats = load_decode_window_stats(metrics_dir)
        if dw_stats and dw_stats.get("count", 0) > 0:
            stats = dw_stats
        else:
            all_tbc = [v for r in records for v in r.get("tbc", [])]
            if not all_tbc:
                continue
            stats = compute_stats(all_tbc)

        rows.append((batch_size, input_length, stats))

    if not rows:
        return

    rows.sort(key=lambda r: (r[0], r[1]))

    table = Table(title="Decode Results (TBT in decode window)")
    table.add_column("Batch Size", justify="right", style="cyan")
    table.add_column("Input Length", justify="right", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")
    table.add_column("Min (ms)", justify="right", style="dim")
    table.add_column("Max (ms)", justify="right", style="dim")
    table.add_column("Samples", justify="right", style="dim")

    for bs, il, stats in rows:
        table.add_row(
            str(bs),
            str(il),
            fmt_ms(stats.get("mean")),
            fmt_ms(stats.get("median")),
            fmt_ms(stats.get("p99")),
            fmt_ms(stats.get("min")),
            fmt_ms(stats.get("max")),
            str(stats.get("count", "—")),
        )

    console.print()
    console.print(table)
    console.print()

    save_json(
        {
            "type": "decode",
            "metric": "tbt",
            "results": [
                {"batch_size": bs, "input_length": il, "tbt": s} for bs, il, s in rows
            ],
        },
        os.path.join(cfg.output_dir, "decode_results.json"),
    )

    # CSV
    csv_path = os.path.join(cfg.output_dir, "decode_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_size",
                "input_length",
                "mean_ms",
                "p50_ms",
                "p99_ms",
                "min_ms",
                "max_ms",
                "samples",
            ],
        )
        writer.writeheader()
        for bs, il, s in rows:
            writer.writerow(
                {
                    "batch_size": bs,
                    "input_length": il,
                    "mean_ms": s.get("mean", 0) * 1000,
                    "p50_ms": s.get("median", 0) * 1000,
                    "p99_ms": s.get("p99", 0) * 1000,
                    "min_ms": s.get("min", 0) * 1000,
                    "max_ms": s.get("max", 0) * 1000,
                    "samples": s.get("count", 0),
                }
            )
    console.print(f"  CSV saved to {csv_path}")

    # Plots: TBT vs batch size, one line per input length
    if len(rows) >= 2:
        plots_dir = os.path.join(cfg.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Group by input_length
        by_il: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        for bs, il, s in rows:
            by_il[il].append((bs, s))

        # TBT P50 vs batch size
        fig, ax = plt.subplots(figsize=(8, 5))
        for il in sorted(by_il.keys()):
            points = sorted(by_il[il], key=lambda x: x[0])
            ax.plot(
                [p[0] for p in points],
                [p[1].get("median", 0) * 1000 for p in points],
                "o-",
                label=f"il={il}",
                linewidth=2,
            )
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("TBT P50 (ms)")
        ax.set_title("Time Between Tokens (P50) vs Batch Size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "tbt_p50_vs_batch_size.png"), dpi=150)
        plt.close(fig)

        # TBT P99 vs batch size
        fig, ax = plt.subplots(figsize=(8, 5))
        for il in sorted(by_il.keys()):
            points = sorted(by_il[il], key=lambda x: x[0])
            ax.plot(
                [p[0] for p in points],
                [p[1].get("p99", 0) * 1000 for p in points],
                "s--",
                label=f"il={il}",
                linewidth=2,
            )
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("TBT P99 (ms)")
        ax.set_title("Time Between Tokens (P99) vs Batch Size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "tbt_p99_vs_batch_size.png"), dpi=150)
        plt.close(fig)

        console.print(f"  Plots saved to {plots_dir}/")

    console.print()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def run_decode(cfg: DecodeMicrobenchmarkConfig) -> None:
    """Run a single decode microbenchmark."""
    from veeksha.microbench.runner import run

    run(cfg, "decode", BANNER_ROWS, build_benchmark_configs, print_results_table, validate)


def main() -> None:
    for cfg in DecodeMicrobenchmarkConfig.create_from_cli_args():
        run_decode(cfg)
