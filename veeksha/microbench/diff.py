"""Diff: generate comparative plots from two microbenchmark output directories."""

import json
import os
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vidhi import field, frozen_dataclass

from veeksha.cli.base import VeekshaCommand
from veeksha.microbench.common import console

# ---------------------------------------------------------------------------
# Detect benchmark type from results JSON
# ---------------------------------------------------------------------------

_TYPE_FILES = {
    "prefill": "prefill_results.json",
    "decode": "decode_results.json",
    "stress": "stress_results.json",
}


def _detect_type(output_dir: str) -> tuple[str, dict] | None:
    """Detect benchmark type and load results JSON from an output directory."""
    base = Path(output_dir)
    for bench_type, filename in _TYPE_FILES.items():
        candidates = sorted(base.glob(f"**/{filename}"))
        if candidates:
            with open(candidates[-1]) as f:
                data = json.load(f)
            return bench_type, data
    return None


# ---------------------------------------------------------------------------
# Prefill comparative plots
# ---------------------------------------------------------------------------


def _plot_prefill(data1: dict, data2: dict, label1: str, label2: str, out: str) -> None:
    """Generate comparative prefill plots (TTFC vs input length)."""
    rows1 = data1.get("results", [])
    rows2 = data2.get("results", [])
    if not rows1 or not rows2:
        return

    os.makedirs(out, exist_ok=True)

    def _extract(rows: list[dict]) -> tuple[list[int], list[float], list[float]]:
        lengths = [r["input_length"] for r in rows]
        p50s = [r["ttfc"]["median"] * 1000 for r in rows]
        p99s = [r["ttfc"]["p99"] * 1000 for r in rows]
        return lengths, p50s, p99s

    il1, p50_1, p99_1 = _extract(rows1)
    il2, p50_2, p99_2 = _extract(rows2)

    # TTFC P50 comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(il1, p50_1, "o-", label=f"{label1}", linewidth=2)
    ax.plot(il2, p50_2, "s--", label=f"{label2}", linewidth=2)
    ax.set_xlabel("Input Length (tokens)")
    ax.set_ylabel("TTFC P50 (ms)")
    ax.set_title("Time to First Token (P50) vs Input Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ttfc_p50_vs_input_length.png"), dpi=150)
    plt.close(fig)

    # TTFC P99 comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(il1, p99_1, "o-", label=f"{label1}", linewidth=2)
    ax.plot(il2, p99_2, "s--", label=f"{label2}", linewidth=2)
    ax.set_xlabel("Input Length (tokens)")
    ax.set_ylabel("TTFC P99 (ms)")
    ax.set_title("Time to First Token (P99) vs Input Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ttfc_p99_vs_input_length.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Decode comparative plots
# ---------------------------------------------------------------------------


def _plot_decode(data1: dict, data2: dict, label1: str, label2: str, out: str) -> None:
    """Generate comparative decode plots (TBT vs batch size per input length)."""
    rows1 = data1.get("results", [])
    rows2 = data2.get("results", [])
    if not rows1 or not rows2:
        return

    os.makedirs(out, exist_ok=True)

    def _group_by_il(
        rows: list[dict],
    ) -> dict[int, list[tuple[int, dict]]]:
        by_il: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        for r in rows:
            by_il[r["input_length"]].append((r["batch_size"], r["tbt"]))
        return by_il

    by_il1 = _group_by_il(rows1)
    by_il2 = _group_by_il(rows2)
    common_ils = sorted(set(by_il1.keys()) & set(by_il2.keys()))

    if not common_ils:
        console.print("  [yellow]No common input lengths to compare for decode.[/]")
        return

    # TBT P50 comparison per input length
    fig, ax = plt.subplots(figsize=(8, 5))
    for il in common_ils:
        pts1 = sorted(by_il1[il], key=lambda x: x[0])
        pts2 = sorted(by_il2[il], key=lambda x: x[0])
        ax.plot(
            [p[0] for p in pts1],
            [p[1].get("median", 0) * 1000 for p in pts1],
            "o-",
            label=f"{label1} il={il}",
            linewidth=2,
        )
        ax.plot(
            [p[0] for p in pts2],
            [p[1].get("median", 0) * 1000 for p in pts2],
            "s--",
            label=f"{label2} il={il}",
            linewidth=2,
        )
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("TBT P50 (ms)")
    ax.set_title("Time Between Tokens (P50) vs Batch Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tbt_p50_vs_batch_size.png"), dpi=150)
    plt.close(fig)

    # TBT P99 comparison per input length
    fig, ax = plt.subplots(figsize=(8, 5))
    for il in common_ils:
        pts1 = sorted(by_il1[il], key=lambda x: x[0])
        pts2 = sorted(by_il2[il], key=lambda x: x[0])
        ax.plot(
            [p[0] for p in pts1],
            [p[1].get("p99", 0) * 1000 for p in pts1],
            "o-",
            label=f"{label1} il={il}",
            linewidth=2,
        )
        ax.plot(
            [p[0] for p in pts2],
            [p[1].get("p99", 0) * 1000 for p in pts2],
            "s--",
            label=f"{label2} il={il}",
            linewidth=2,
        )
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("TBT P99 (ms)")
    ax.set_title("Time Between Tokens (P99) vs Batch Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tbt_p99_vs_batch_size.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stress comparative plots
# ---------------------------------------------------------------------------


def _plot_stress(data1: dict, data2: dict, label1: str, label2: str, out: str) -> None:
    """Generate comparative stress plots."""
    results1 = data1.get("results", [])
    results2 = data2.get("results", [])
    if not results1 or not results2:
        return

    os.makedirs(out, exist_ok=True)

    # Determine level label from traffic mode (use first run's mode)
    traffic_mode = data1.get("traffic_mode", "")
    level_label = "QPS" if "fixed-rate" in traffic_mode else "Concurrency"

    levels1 = [r["level"] for r in results1]
    levels2 = [r["level"] for r in results2]

    # 1. Throughput vs Load (input + output)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        levels1,
        [r["input_throughput"] for r in results1],
        "o-",
        label=f"{label1} input",
        linewidth=2,
    )
    ax.plot(
        levels1,
        [r["output_throughput"] for r in results1],
        "o--",
        label=f"{label1} output",
        linewidth=2,
    )
    ax.plot(
        levels2,
        [r["input_throughput"] for r in results2],
        "s-",
        label=f"{label2} input",
        linewidth=2,
    )
    ax.plot(
        levels2,
        [r["output_throughput"] for r in results2],
        "s--",
        label=f"{label2} output",
        linewidth=2,
    )
    ax.set_xlabel(level_label)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Throughput vs Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "throughput_vs_load.png"), dpi=150)
    plt.close(fig)

    # 2. E2E Latency vs Load (P50 + P99)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        levels1,
        [r["e2e_latency_p50"] * 1000 for r in results1],
        "o-",
        label=f"{label1} P50",
        linewidth=2,
    )
    ax.plot(
        levels1,
        [r["e2e_latency_p99"] * 1000 for r in results1],
        "o--",
        label=f"{label1} P99",
        linewidth=2,
    )
    ax.plot(
        levels2,
        [r["e2e_latency_p50"] * 1000 for r in results2],
        "s-",
        label=f"{label2} P50",
        linewidth=2,
    )
    ax.plot(
        levels2,
        [r["e2e_latency_p99"] * 1000 for r in results2],
        "s--",
        label=f"{label2} P99",
        linewidth=2,
    )
    ax.set_xlabel(level_label)
    ax.set_ylabel("E2E Latency (ms)")
    ax.set_title("E2E Latency vs Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "e2e_latency_vs_load.png"), dpi=150)
    plt.close(fig)

    # 3. E2E Latency vs Output Throughput (tradeoff curve)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [r["e2e_latency_p50"] * 1000 for r in results1],
        [r["output_throughput"] for r in results1],
        "o-",
        label=f"{label1} P50",
        linewidth=2,
    )
    ax.plot(
        [r["e2e_latency_p99"] * 1000 for r in results1],
        [r["output_throughput"] for r in results1],
        "o--",
        label=f"{label1} P99",
        linewidth=2,
    )
    ax.plot(
        [r["e2e_latency_p50"] * 1000 for r in results2],
        [r["output_throughput"] for r in results2],
        "s-",
        label=f"{label2} P50",
        linewidth=2,
    )
    ax.plot(
        [r["e2e_latency_p99"] * 1000 for r in results2],
        [r["output_throughput"] for r in results2],
        "s--",
        label=f"{label2} P99",
        linewidth=2,
    )
    ax.set_xlabel("E2E Latency (ms)")
    ax.set_ylabel("Output Throughput (tok/s)")
    ax.set_title("Output Throughput vs Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "output_throughput_vs_latency.png"), dpi=150)
    plt.close(fig)

    # 4. TTFC vs Load (P50 + P99)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        levels1,
        [r["ttfc_p50"] * 1000 for r in results1],
        "o-",
        label=f"{label1} P50",
        linewidth=2,
    )
    ax.plot(
        levels1,
        [r["ttfc_p99"] * 1000 for r in results1],
        "o--",
        label=f"{label1} P99",
        linewidth=2,
    )
    ax.plot(
        levels2,
        [r["ttfc_p50"] * 1000 for r in results2],
        "s-",
        label=f"{label2} P50",
        linewidth=2,
    )
    ax.plot(
        levels2,
        [r["ttfc_p99"] * 1000 for r in results2],
        "s--",
        label=f"{label2} P99",
        linewidth=2,
    )
    ax.set_xlabel(level_label)
    ax.set_ylabel("TTFC (ms)")
    ax.set_title("Time to First Token vs Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ttfc_vs_load.png"), dpi=150)
    plt.close(fig)

    # 5. Interactivity vs Load (P50 + P99)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        levels1,
        [r["interactivity_p50"] for r in results1],
        "o-",
        label=f"{label1} P50",
        linewidth=2,
    )
    ax.plot(
        levels1,
        [r["interactivity_p99"] for r in results1],
        "o--",
        label=f"{label1} P99",
        linewidth=2,
    )
    ax.plot(
        levels2,
        [r["interactivity_p50"] for r in results2],
        "s-",
        label=f"{label2} P50",
        linewidth=2,
    )
    ax.plot(
        levels2,
        [r["interactivity_p99"] for r in results2],
        "s--",
        label=f"{label2} P99",
        linewidth=2,
    )
    ax.set_xlabel(level_label)
    ax.set_ylabel("Interactivity (tok/s/user)")
    ax.set_title("Interactivity vs Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "interactivity_vs_load.png"), dpi=150)
    plt.close(fig)

    # 6. Interactivity vs Input Throughput
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [r["interactivity_p50"] for r in results1],
        [r["input_throughput"] for r in results1],
        "o-",
        label=f"{label1} P50",
        linewidth=2,
    )
    ax.plot(
        [r["interactivity_p99"] for r in results1],
        [r["input_throughput"] for r in results1],
        "o--",
        label=f"{label1} P99",
        linewidth=2,
    )
    ax.plot(
        [r["interactivity_p50"] for r in results2],
        [r["input_throughput"] for r in results2],
        "s-",
        label=f"{label2} P50",
        linewidth=2,
    )
    ax.plot(
        [r["interactivity_p99"] for r in results2],
        [r["input_throughput"] for r in results2],
        "s--",
        label=f"{label2} P99",
        linewidth=2,
    )
    ax.set_xlabel("Interactivity (tok/s/user)")
    ax.set_ylabel("Input Throughput (tok/s)")
    ax.set_title("Input Throughput vs Interactivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "input_throughput_vs_interactivity.png"), dpi=150)
    plt.close(fig)

    # 7. Interactivity vs Output Throughput
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [r["interactivity_p50"] for r in results1],
        [r["output_throughput"] for r in results1],
        "o-",
        label=f"{label1} P50",
        linewidth=2,
    )
    ax.plot(
        [r["interactivity_p99"] for r in results1],
        [r["output_throughput"] for r in results1],
        "o--",
        label=f"{label1} P99",
        linewidth=2,
    )
    ax.plot(
        [r["interactivity_p50"] for r in results2],
        [r["output_throughput"] for r in results2],
        "s-",
        label=f"{label2} P50",
        linewidth=2,
    )
    ax.plot(
        [r["interactivity_p99"] for r in results2],
        [r["output_throughput"] for r in results2],
        "s--",
        label=f"{label2} P99",
        linewidth=2,
    )
    ax.set_xlabel("Interactivity (tok/s/user)")
    ax.set_ylabel("Output Throughput (tok/s)")
    ax.set_title("Output Throughput vs Interactivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "output_throughput_vs_interactivity.png"), dpi=150)
    plt.close(fig)

    # 8. TPS/GPU vs Load (if either run has per-GPU data)
    has_tps1 = any(r.get("output_tps_per_gpu", 0) > 0 for r in results1)
    has_tps2 = any(r.get("output_tps_per_gpu", 0) > 0 for r in results2)
    if has_tps1 or has_tps2:
        fig, ax = plt.subplots(figsize=(8, 5))
        if has_tps1:
            ax.plot(
                levels1,
                [r.get("output_tps_per_gpu", 0) for r in results1],
                "o-",
                label=f"{label1}",
                linewidth=2,
            )
        if has_tps2:
            ax.plot(
                levels2,
                [r.get("output_tps_per_gpu", 0) for r in results2],
                "s--",
                label=f"{label2}",
                linewidth=2,
            )
        ax.set_xlabel(level_label)
        ax.set_ylabel("TPS / GPU (tok/s/gpu)")
        ax.set_title("Output Throughput per GPU vs Load")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "tps_per_gpu_vs_load.png"), dpi=150)
        plt.close(fig)

        # 9. TPS/GPU vs TPS/User comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        if has_tps1:
            ax.plot(
                [r["interactivity_p50"] for r in results1],
                [r.get("output_tps_per_gpu", 0) for r in results1],
                "o-",
                label=f"{label1}",
                linewidth=2,
                markersize=8,
            )
        if has_tps2:
            ax.plot(
                [r["interactivity_p50"] for r in results2],
                [r.get("output_tps_per_gpu", 0) for r in results2],
                "s--",
                label=f"{label2}",
                linewidth=2,
                markersize=8,
            )
        ax.set_xlabel("TPS / User (tok/s/user)")
        ax.set_ylabel("TPS / GPU (tok/s/gpu)")
        ax.set_title("Throughput Curve: TPS/GPU vs TPS/User")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "tps_per_gpu_vs_tps_per_user.png"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_PLOTTERS = {
    "prefill": _plot_prefill,
    "decode": _plot_decode,
    "stress": _plot_stress,
}


_TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"


def _make_diff_run_dir(cfg: "DiffConfig") -> "DiffConfig":
    """Create a timestamped run directory under output_dir/plot_diff/ and update 'latest' symlink."""
    timestamp = datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT)
    type_dir = os.path.join(cfg.output_dir, "plot_diff")
    run_dir = os.path.join(type_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    latest = os.path.join(type_dir, "latest")
    tmp_link = latest + ".tmp"
    os.symlink(timestamp, tmp_link)
    os.replace(tmp_link, latest)

    return replace(cfg, output_dir=run_dir)


def run_diff(cfg: "DiffConfig") -> None:
    """Run the diff command: detect types, validate, and generate comparative plots."""
    if not cfg.output_dir1 or not cfg.output_dir2:
        sys.exit("Both --output_dir1 and --output_dir2 are required.")

    result1 = _detect_type(cfg.output_dir1)
    if result1 is None:
        sys.exit(
            f"Could not detect benchmark type in {cfg.output_dir1}. "
            "Expected prefill_results.json, decode_results.json, or stress_results.json."
        )

    result2 = _detect_type(cfg.output_dir2)
    if result2 is None:
        sys.exit(
            f"Could not detect benchmark type in {cfg.output_dir2}. "
            "Expected prefill_results.json, decode_results.json, or stress_results.json."
        )

    type1, data1 = result1
    type2, data2 = result2

    if type1 != type2:
        sys.exit(
            f"Benchmark type mismatch: {cfg.output_dir1} is '{type1}' "
            f"but {cfg.output_dir2} is '{type2}'. "
            "Both directories must contain the same benchmark type."
        )

    cfg = _make_diff_run_dir(cfg)

    label1 = cfg.label1 or os.path.basename(os.path.normpath(cfg.output_dir1))
    label2 = cfg.label2 or os.path.basename(os.path.normpath(cfg.output_dir2))

    plots_dir = os.path.join(cfg.output_dir, "plots")
    console.print(f"\n  Benchmark type: [cyan]{type1}[/]")
    console.print(f"  Comparing: [green]{label1}[/] vs [green]{label2}[/]")

    plotter = _PLOTTERS[type1]
    plotter(data1, data2, label1, label2, plots_dir)

    console.print(f"  Comparative plots saved to {plots_dir}/\n")


# ---------------------------------------------------------------------------
# Config (imported by commands.py)
# ---------------------------------------------------------------------------


@frozen_dataclass
class DiffConfig(VeekshaCommand, name="diff"):
    """Compare two benchmark output directories with side-by-side plots."""

    output_dir1: str = field("", help="First benchmark output directory")
    output_dir2: str = field("", help="Second benchmark output directory")
    output_dir: str = field(
        "microbench_output", help="Output directory for comparative plots"
    )
    label1: str = field("", help="Label for first run (defaults to directory name)")
    label2: str = field("", help="Label for second run (defaults to directory name)")
