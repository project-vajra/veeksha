"""Stress microbenchmark: throughput-vs-latency curves at increasing concurrency."""

import csv
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, fields

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.table import Table

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.config.generator.channel import TextChannelGeneratorConfig
from veeksha.config.generator.interval import PoissonIntervalGeneratorConfig
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
from veeksha.config.traffic import ConcurrentTrafficConfig, RateTrafficConfig
from veeksha.microbench.common import (
    ValidationResult,
    build_client_config,
    console,
    find_all_run_metrics,
    fmt_ms,
    load_request_metrics,
    percentile,
    save_json,
)
from veeksha.microbench.config import (
    AutoStressConfig,
    ManualStressConfig,
    RangeStressConfig,
    StressMicrobenchmarkConfig,
    StressTrafficMode,
)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER_ROWS: list[tuple[str, str]] = [
    ("Input length", "input_length"),
    ("Output length", "output_length"),
    ("Traffic mode", "traffic_mode"),
    ("Point duration", "point_duration"),
    ("Warmup duration", "warmup_duration"),
]


# ---------------------------------------------------------------------------
# Concurrency / QPS level resolution
# ---------------------------------------------------------------------------


def resolve_levels(cfg: StressMicrobenchmarkConfig) -> list[int]:
    """Resolve load levels for manual/range modes."""
    if isinstance(cfg, ManualStressConfig):
        return sorted(set(cfg.concurrency_levels))
    if isinstance(cfg, RangeStressConfig):
        return _log_spaced_levels(
            cfg.concurrency_min, cfg.concurrency_max, cfg.concurrency_points
        )
    raise ValueError(f"Cannot resolve levels for {type(cfg).__name__}")


def _log_spaced_levels(lo: int, hi: int, n: int) -> list[int]:
    """Generate n log-spaced integer levels in [lo, hi], deduplicated."""
    if n <= 1:
        return [lo]
    log_lo = math.log(max(lo, 1))
    log_hi = math.log(max(hi, 1))
    raw = [math.exp(log_lo + (log_hi - log_lo) * i / (n - 1)) for i in range(n)]
    return sorted(set(max(1, round(v)) for v in raw))


# ---------------------------------------------------------------------------
# Session budget
# ---------------------------------------------------------------------------


def estimate_max_sessions(
    level: int,
    duration: int,
    output_length: int,
    max_tps: float,
    traffic_mode: StressTrafficMode,
) -> int:
    """Estimate sessions needed so the benchmark never runs out."""
    if traffic_mode == StressTrafficMode.FIXED_RATE:
        estimated = level * duration * 2
    else:
        estimated = level * duration * max_tps / output_length
    return int(max(estimated * 2, level * 10))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _build_traffic_config(
    cfg: StressMicrobenchmarkConfig, level: int
) -> ConcurrentTrafficConfig | RateTrafficConfig:
    """Build traffic config based on traffic_mode."""
    if cfg.traffic_mode == StressTrafficMode.FIXED_RATE:
        return RateTrafficConfig(
            interval_generator=PoissonIntervalGeneratorConfig(
                arrival_rate=float(level)
            ),
            cancel_session_on_failure=False,
        )
    return ConcurrentTrafficConfig(
        target_concurrent_sessions=level,
        rampup_seconds=0,
        cancel_session_on_failure=False,
    )


def _build_one_config(cfg: StressMicrobenchmarkConfig, level: int) -> BenchmarkConfig:
    """Build a BenchmarkConfig for a single load level."""
    max_sessions = estimate_max_sessions(
        level,
        cfg.point_duration,
        cfg.output_length,
        cfg.max_tokens_per_second_estimate,
        cfg.traffic_mode,
    )
    return BenchmarkConfig(
        output_dir=f"{cfg.output_dir}/c={level}",
        seed=cfg.seed,
        session_generator=SyntheticSessionGeneratorConfig(
            session_graph=SingleRequestSessionGraphGeneratorConfig(),
            channels=[
                TextChannelGeneratorConfig(
                    body_length_generator=FixedLengthGeneratorConfig(
                        value=cfg.input_length,
                    ),
                )
            ],
            output_spec=OutputSpecConfig(
                text=TextOutputSpecConfig(
                    output_length_generator=FixedLengthGeneratorConfig(
                        value=cfg.output_length,
                    ),
                ),
            ),
        ),
        traffic_scheduler=_build_traffic_config(cfg, level),
        evaluators=[
            PerformanceEvaluatorConfig(stream_metrics=False, slos=[]),
        ],
        client=build_client_config(cfg),
        runtime=RuntimeConfig(
            max_sessions=max_sessions,
            benchmark_timeout=cfg.point_duration,
            num_client_threads=max(level, 3),
            pregenerate_sessions=True,
        ),
        trace_recorder=TraceRecorderConfig(enabled=False),
    )


def build_benchmark_configs(
    cfg: StressMicrobenchmarkConfig,
) -> list[BenchmarkConfig]:
    """Build configs for all load levels (manual/range modes)."""
    levels = resolve_levels(cfg)
    return [_build_one_config(cfg, c) for c in levels]


# ---------------------------------------------------------------------------
# Stress point result
# ---------------------------------------------------------------------------


@dataclass
class StressPointResult:
    """Metrics extracted from a single load level."""

    level: int
    input_throughput: float  # total input tok/s (system-level)
    output_throughput: float  # total output tok/s (system-level)
    e2e_latency_p50: float
    e2e_latency_p99: float
    ttfc_p50: float
    ttfc_p99: float
    tpot_p50: float
    tpot_p99: float
    interactivity_p50: float  # 1 / tpot_p50 (tok/s/user)
    interactivity_p99: float  # 1 / tpot_p99 (tok/s/user)
    num_requests: int


def _extract_stress_point(
    level: int,
    metrics: list[dict],
    warmup_seconds: int,
    input_length: int,
) -> StressPointResult | None:
    """Extract stress metrics from raw request-level metrics."""
    if not metrics:
        return None

    # Filter warmup: discard requests completing before min(dispatched) + warmup
    min_dispatch = min(r["scheduler_dispatched_at"] for r in metrics)
    cutoff = min_dispatch + warmup_seconds
    post_warmup = [r for r in metrics if r["client_completed_at"] > cutoff]

    if len(post_warmup) < 2:
        return None

    # System throughput: tokens / wall-clock span
    total_output = sum(r["num_output_tokens"] for r in post_warmup)
    total_input = sum(
        r.get("target_num_delta_prompt_tokens", input_length) for r in post_warmup
    )
    first_dispatch = min(r["scheduler_dispatched_at"] for r in post_warmup)
    last_complete = max(r["client_completed_at"] for r in post_warmup)
    span = last_complete - first_dispatch
    output_throughput = total_output / span if span > 0 else 0.0
    input_throughput = total_input / span if span > 0 else 0.0

    # Latency stats
    e2e = [r["end_to_end_latency"] for r in post_warmup if r.get("end_to_end_latency")]
    ttfc_vals = [r["ttfc"] for r in post_warmup if r.get("ttfc")]
    tpot_vals = [r["tpot"] for r in post_warmup if r.get("tpot") and r["tpot"] > 0]

    tpot_p50 = percentile(tpot_vals, 50) if tpot_vals else 0.0
    tpot_p99 = percentile(tpot_vals, 99) if tpot_vals else 0.0

    return StressPointResult(
        level=level,
        input_throughput=input_throughput,
        output_throughput=output_throughput,
        e2e_latency_p50=percentile(e2e, 50) if e2e else 0.0,
        e2e_latency_p99=percentile(e2e, 99) if e2e else 0.0,
        ttfc_p50=percentile(ttfc_vals, 50) if ttfc_vals else 0.0,
        ttfc_p99=percentile(ttfc_vals, 99) if ttfc_vals else 0.0,
        tpot_p50=tpot_p50,
        tpot_p99=tpot_p99,
        interactivity_p50=1.0 / tpot_p50 if tpot_p50 > 0 else 0.0,
        interactivity_p99=1.0 / tpot_p99 if tpot_p99 > 0 else 0.0,
        num_requests=len(post_warmup),
    )


# ---------------------------------------------------------------------------
# Results: table, JSON, CSV, plots
# ---------------------------------------------------------------------------


def _parse_level_from_dir(dirname: str) -> int | None:
    """Extract level from directory name like 'c=16'."""
    m = re.search(r"c=(\d+)", dirname)
    return int(m.group(1)) if m else None


def _collect_results(
    cfg: StressMicrobenchmarkConfig,
) -> list[StressPointResult]:
    """Collect and sort stress point results from all run directories."""
    all_runs = find_all_run_metrics(cfg.output_dir)
    if not all_runs:
        return []

    results: list[StressPointResult] = []
    for metrics_dir, records in all_runs:
        level = _parse_level_from_dir(str(metrics_dir))
        if level is None:
            continue
        point = _extract_stress_point(
            level, records, cfg.warmup_duration, cfg.input_length
        )
        if point is not None:
            results.append(point)

    results.sort(key=lambda r: r.level)
    return results


def _save_csv(results: list[StressPointResult], path: str) -> None:
    """Save all metrics as CSV for downstream use."""
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [f.name for f in fields(StressPointResult)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def _save_plots(
    results: list[StressPointResult],
    output_dir: str,
    level_label: str,
) -> None:
    """Generate and save stress benchmark plots."""
    if len(results) < 2:
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    levels = [r.level for r in results]
    in_tputs = [r.input_throughput for r in results]
    out_tputs = [r.output_throughput for r in results]

    # 1. Throughput vs Load Level (input + output)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(levels, in_tputs, "o-", label="Input", linewidth=2)
    ax.plot(levels, out_tputs, "s--", label="Output", linewidth=2)
    ax.set_xlabel(level_label)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Throughput vs Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "throughput_vs_load.png"), dpi=150)
    plt.close(fig)

    # 2. E2E Latency vs Load Level (P50 + P99)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        levels,
        [r.e2e_latency_p50 * 1000 for r in results],
        "o-",
        label="P50",
        linewidth=2,
    )
    ax.plot(
        levels,
        [r.e2e_latency_p99 * 1000 for r in results],
        "s--",
        label="P99",
        linewidth=2,
    )
    ax.set_xlabel(level_label)
    ax.set_ylabel("E2E Latency (ms)")
    ax.set_title("E2E Latency vs Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "e2e_latency_vs_load.png"), dpi=150)
    plt.close(fig)

    # 3. E2E Latency vs Output Throughput (the tradeoff curve)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [r.e2e_latency_p50 * 1000 for r in results],
        out_tputs,
        "o-",
        label="P50",
        linewidth=2,
    )
    ax.plot(
        [r.e2e_latency_p99 * 1000 for r in results],
        out_tputs,
        "s--",
        label="P99",
        linewidth=2,
    )
    ax.set_xlabel("E2E Latency (ms)")
    ax.set_ylabel("Output Throughput (tok/s)")
    ax.set_title("Output Throughput vs Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "output_throughput_vs_latency.png"), dpi=150)
    plt.close(fig)

    # 4. TTFC vs Load Level (P50 + P99)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        levels, [r.ttfc_p50 * 1000 for r in results], "o-", label="P50", linewidth=2
    )
    ax.plot(
        levels, [r.ttfc_p99 * 1000 for r in results], "s--", label="P99", linewidth=2
    )
    ax.set_xlabel(level_label)
    ax.set_ylabel("TTFC (ms)")
    ax.set_title("Time to First Token vs Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "ttfc_vs_load.png"), dpi=150)
    plt.close(fig)

    # 5. Interactivity vs Load Level (P50 + P99)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        levels, [r.interactivity_p50 for r in results], "o-", label="P50", linewidth=2
    )
    ax.plot(
        levels, [r.interactivity_p99 for r in results], "s--", label="P99", linewidth=2
    )
    ax.set_xlabel(level_label)
    ax.set_ylabel("Interactivity (tok/s/user)")
    ax.set_title("Interactivity vs Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "interactivity_vs_load.png"), dpi=150)
    plt.close(fig)

    # 6. Interactivity vs Input Throughput (P50 + P99)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [r.interactivity_p50 for r in results], in_tputs, "o-", label="P50", linewidth=2
    )
    ax.plot(
        [r.interactivity_p99 for r in results],
        in_tputs,
        "s--",
        label="P99",
        linewidth=2,
    )
    ax.set_xlabel("Interactivity (tok/s/user)")
    ax.set_ylabel("Input Throughput (tok/s)")
    ax.set_title("Input Throughput vs Interactivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(plots_dir, "input_throughput_vs_interactivity.png"), dpi=150
    )
    plt.close(fig)

    # 7. Interactivity vs Output Throughput (P50 + P99)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [r.interactivity_p50 for r in results],
        out_tputs,
        "o-",
        label="P50",
        linewidth=2,
    )
    ax.plot(
        [r.interactivity_p99 for r in results],
        out_tputs,
        "s--",
        label="P99",
        linewidth=2,
    )
    ax.set_xlabel("Interactivity (tok/s/user)")
    ax.set_ylabel("Output Throughput (tok/s)")
    ax.set_title("Output Throughput vs Interactivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(plots_dir, "output_throughput_vs_interactivity.png"), dpi=150
    )
    plt.close(fig)


def print_results_table(cfg: StressMicrobenchmarkConfig) -> None:
    """Print stress results table and save JSON, CSV, and plots."""
    results = _collect_results(cfg)
    if not results:
        return

    level_label = (
        "QPS" if cfg.traffic_mode == StressTrafficMode.FIXED_RATE else "Concurrency"
    )

    table = Table(title="Stress Results (Throughput vs Latency)")
    table.add_column(level_label, justify="right", style="cyan")
    table.add_column("In Tput\n(tok/s)", justify="right")
    table.add_column("Out Tput\n(tok/s)", justify="right")
    table.add_column("E2E P50\n(ms)", justify="right")
    table.add_column("E2E P99\n(ms)", justify="right")
    table.add_column("TTFC P50\n(ms)", justify="right")
    table.add_column("TTFC P99\n(ms)", justify="right")
    table.add_column("Intrctvty\nP50 (t/s)", justify="right")
    table.add_column("Intrctvty\nP99 (t/s)", justify="right")
    table.add_column("Reqs", justify="right", style="dim")

    for r in results:
        table.add_row(
            str(r.level),
            f"{r.input_throughput:.1f}",
            f"{r.output_throughput:.1f}",
            fmt_ms(r.e2e_latency_p50),
            fmt_ms(r.e2e_latency_p99),
            fmt_ms(r.ttfc_p50),
            fmt_ms(r.ttfc_p99),
            f"{r.interactivity_p50:.1f}",
            f"{r.interactivity_p99:.1f}",
            str(r.num_requests),
        )

    console.print()
    console.print(table)
    console.print()

    # JSON
    save_json(
        {
            "type": "stress",
            "traffic_mode": str(cfg.traffic_mode),
            "input_length": cfg.input_length,
            "output_length": cfg.output_length,
            "results": [asdict(r) for r in results],
        },
        os.path.join(cfg.output_dir, "stress_results.json"),
    )

    # CSV
    csv_path = os.path.join(cfg.output_dir, "stress_results.csv")
    _save_csv(results, csv_path)
    console.print(f"  CSV saved to {csv_path}")

    # Plots
    _save_plots(results, cfg.output_dir, level_label)
    console.print(f"  Plots saved to {os.path.join(cfg.output_dir, 'plots')}/")
    console.print()


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate(cfg: StressMicrobenchmarkConfig, output_dir: str) -> ValidationResult:
    """Validate stress benchmark results."""
    result = ValidationResult()
    all_runs = find_all_run_metrics(output_dir)

    if not all_runs:
        result.fail("metrics_found", "No request_level_metrics.jsonl found")
        return result

    points: list[StressPointResult] = []
    for metrics_dir, records in all_runs:
        level = _parse_level_from_dir(str(metrics_dir))
        if level is None:
            continue

        label = f"c={level}"

        # Check for failed requests
        failed = [r for r in records if r.get("num_output_tokens", 0) == 0]
        if failed:
            result.fail(
                f"no_errors [{label}]",
                f"{len(failed)} requests produced 0 output tokens",
            )
        else:
            result.passed(f"no_errors [{label}]")

        # Extract point and check sample count
        point = _extract_stress_point(
            level, records, cfg.warmup_duration, cfg.input_length
        )
        if point is None or point.num_requests < 10:
            count = point.num_requests if point else 0
            result.warn(
                f"sample_count [{label}]",
                f"only {count} requests after warmup (need >= 10)",
            )
        else:
            result.passed(f"sample_count [{label}]", f"{point.num_requests} requests")
            points.append(point)

    # Throughput monotonicity check (warn only)
    points.sort(key=lambda p: p.level)
    if len(points) >= 2:
        drops = 0
        for i in range(1, len(points)):
            if points[i].output_throughput < points[i - 1].output_throughput * 0.9:
                drops += 1
        if drops == 0:
            result.passed(
                "throughput_monotonicity",
                "throughput is non-decreasing across load levels",
            )
        else:
            result.warn(
                "throughput_monotonicity",
                f"throughput dropped significantly at {drops} point(s) — "
                "may indicate saturation or instability",
            )

    return result


# ---------------------------------------------------------------------------
# Auto mode
# ---------------------------------------------------------------------------


def _load_existing_point(
    cfg: StressMicrobenchmarkConfig, level: int
) -> StressPointResult | None:
    """Try to load results from an already-completed c=N directory."""
    metrics = load_request_metrics(f"{cfg.output_dir}/c={level}")
    if metrics is None:
        return None
    return _extract_stress_point(level, metrics, cfg.warmup_duration, cfg.input_length)


def _run_and_measure(
    cfg: StressMicrobenchmarkConfig, level: int
) -> StressPointResult | None:
    """Run a single load level and extract metrics."""
    from veeksha.cli.benchmarks import BenchmarkCliRunner

    bc = _build_one_config(cfg, level)
    BenchmarkCliRunner([bc]).run_all()

    return _load_existing_point(cfg, level)


def _resume_existing_results(
    cfg: AutoStressConfig,
) -> dict[int, StressPointResult]:
    """Load results from a previous run directory (--resume-dir) by copying c=N dirs."""
    measured: dict[int, StressPointResult] = {}
    resume_dir = cfg.resume_dir
    if not resume_dir or not os.path.isdir(resume_dir):
        return measured

    from veeksha.logger import init_logger

    logger = init_logger(__name__)

    for entry in os.listdir(resume_dir):
        level = _parse_level_from_dir(entry)
        if level is None:
            continue
        src = os.path.realpath(os.path.join(resume_dir, entry))
        dst = os.path.join(cfg.output_dir, entry)
        if os.path.isdir(src) and not os.path.exists(dst):
            os.symlink(src, dst)
            logger.info(f"Resumed c={level} from {src} (symlink)")

        point = _load_existing_point(cfg, level)
        if point is not None:
            measured[level] = point

    if measured:
        logger.info(
            f"Resumed {len(measured)} points from {resume_dir}: "
            f"c={sorted(measured.keys())}"
        )
    return measured


def _find_close_enough(
    target: int, measured: dict[int, StressPointResult], tolerance: float = 0.2
) -> int | None:
    """Find an existing measured level within tolerance of the target.

    Returns the measured level if |measured - target| / target <= tolerance,
    preferring the closest match. Returns None if no match.
    """
    best: int | None = None
    best_dist = float("inf")
    for c in measured:
        dist = abs(c - target) / max(target, 1)
        if dist <= tolerance and dist < best_dist:
            best = c
            best_dist = dist
    return best


def _run_auto_sweep(cfg: AutoStressConfig) -> list[StressPointResult]:
    """Three-phase auto sweep: probe, refine, fill.

    - Reuses results from --resume-dir if provided
    - During fill, skips levels where an existing measurement is close enough
    """
    from veeksha.logger import init_logger

    logger = init_logger(__name__)

    measured: dict[int, StressPointResult] = _resume_existing_results(cfg)

    def _probe(c: int) -> StressPointResult | None:
        if c in measured:
            return measured[c]
        # Check if we already have results on disk (e.g. from a resumed dir)
        point = _load_existing_point(cfg, c)
        if point is None:
            point = _run_and_measure(cfg, c)
        if point is not None:
            measured[c] = point
        return point

    # Phase 1: Exponential probe (upper bound via throughput saturation)
    c = 1
    prev_throughput = 0.0
    probes = 0
    while probes < cfg.auto_max_probes:
        point = _probe(c)
        probes += 1
        if point is None:
            break
        if (
            prev_throughput > 0
            and (point.output_throughput - prev_throughput) / prev_throughput
            < cfg.auto_throughput_threshold
        ):
            break
        prev_throughput = point.output_throughput
        c *= 2

    if not measured:
        return []

    upper = max(measured.keys())

    # Phase 2: Lower bound via interactivity plateau.
    # Walk from lowest concurrency upward; the lower bound is the highest c
    # where interactivity_p50 is still within threshold of the best observed.
    max_interactivity = max(p.interactivity_p50 for p in measured.values())
    interactivity_floor = max_interactivity * (1 - cfg.auto_throughput_threshold)
    sorted_points = sorted(measured.items(), key=lambda kv: kv[0])
    lower = sorted_points[0][0]  # fallback to lowest probed
    for c, p in sorted_points:
        if p.interactivity_p50 >= interactivity_floor:
            lower = c
            break

    logger.info(
        f"Auto sweep bounds: lower={lower} (interactivity plateau), upper={upper} (throughput plateau)"
    )

    # Phase 3: Fill log-spaced levels between lower and upper.
    # Reuse existing measurements that are close enough to a fill target.
    fill_levels = _log_spaced_levels(lower, upper, cfg.auto_fill_points)
    for target_c in fill_levels:
        if target_c in measured:
            continue
        existing = _find_close_enough(target_c, measured)
        if existing is not None:
            logger.info(
                f"Skipping c={target_c}: reusing existing c={existing} "
                f"(within tolerance)"
            )
            continue
        _probe(target_c)

    return sorted(measured.values(), key=lambda p: p.level)


def _run_auto_main(cfg: AutoStressConfig) -> None:
    """Full auto mode entrypoint."""
    from veeksha.microbench.runner import _make_run_dir, _print_banner

    if cfg.resume_dir:
        # When resuming, still create a new timestamped run dir
        # (results get copied in from the resume dir)
        pass
    cfg = _make_run_dir(cfg, "stress")  # type: ignore[assignment]
    _print_banner(cfg, "stress (auto)", BANNER_ROWS)

    _run_auto_sweep(cfg)

    print_results_table(cfg)

    if not cfg.skip_validation:
        result = validate(cfg, cfg.output_dir)
        if result.ok:
            from veeksha.logger import init_logger

            logger = init_logger(__name__)
            num_passed = sum(1 for s, _, _ in result.checks if s == "PASS")
            logger.info(f"Validation passed ({num_passed} checks)")
        else:
            from veeksha.microbench.runner import _print_validation_failure

            _print_validation_failure(result)
            sys.exit(1)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    from argparse import ArgumentParser

    from veeksha.microbench.config import STRESS_MODE_TO_CONFIG

    # Pre-parse --stress-mode to select the right config class
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--stress-mode",
        default="manual",
        choices=list(STRESS_MODE_TO_CONFIG.keys()),
    )
    known, remaining = pre_parser.parse_known_args()

    config_cls = STRESS_MODE_TO_CONFIG[known.stress_mode]

    # Replace sys.argv so create_from_cli_args doesn't see --stress-mode
    sys.argv = [sys.argv[0]] + remaining

    for cfg in config_cls.create_from_cli_args():
        if isinstance(cfg, AutoStressConfig):
            _run_auto_main(cfg)
        else:
            from veeksha.microbench.runner import run

            run(
                cfg,
                "stress",
                BANNER_ROWS,
                build_benchmark_configs,
                print_results_table,
                validate,
            )
