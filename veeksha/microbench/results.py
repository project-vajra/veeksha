"""Post-run results tables for microbenchmarks."""

import json
import os
import statistics
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from veeksha.logger import init_logger
from veeksha.microbench.config import MicrobenchmarkConfig

logger = init_logger(__name__)
console = Console()


def _fmt_ms(val: float | None) -> str:
    """Format a latency value in seconds as milliseconds."""
    if val is None:
        return "—"
    return f"{val * 1000:.2f}"


def _percentile(data: list[float], percentile: float) -> float:
    """Compute a percentile from sorted data (0–100 scale)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    fractional_index = (len(sorted_data) - 1) * (percentile / 100)
    floor_index = int(fractional_index)
    ceil_index = floor_index + 1
    if ceil_index >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[floor_index] + (fractional_index - floor_index) * (sorted_data[ceil_index] - sorted_data[floor_index])


def _compute_stats(values: list[float]) -> dict[str, Any]:
    """Compute summary statistics for a list of values (in seconds)."""
    return {
        "mean": statistics.mean(values),
        "median": _percentile(values, 50),
        "p99": _percentile(values, 99),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def _find_all_run_metrics(base_dir: str) -> list[tuple[Path, list[dict]]]:
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


def _load_decode_window_stats(metrics_dir: Path) -> dict | None:
    """Load decode_window_metrics.json from a metrics directory."""
    path = metrics_dir / "decode_window_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("tbc_in_window_stats")


def _save_json(data: Any, path: str) -> None:
    """Save data as JSON, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def print_results_table(cfg: MicrobenchmarkConfig) -> None:
    """Collect results, print a rich table, and save to JSON."""
    if cfg.type == "prefill":
        _report_prefill(cfg)
    elif cfg.type == "decode":
        _report_decode(cfg)
    elif cfg.type == "mixed":
        _report_mixed(cfg)


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def _collect_prefill_results(cfg: MicrobenchmarkConfig) -> list[dict[str, Any]] | None:
    # Prefill runs directly in cfg.output_dir (single BenchmarkConfig)
    all_runs = _find_all_run_metrics(cfg.output_dir)
    if not all_runs:
        return None

    # All requests are in one run — flatten
    metrics = [record for _, records in all_runs for record in records]

    ttfc_by_input_length: dict[int, list[float]] = {}
    for record in metrics:
        input_length = record["target_num_delta_prompt_tokens"]
        ttfc = record.get("ttfc")
        if ttfc is not None:
            ttfc_by_input_length.setdefault(input_length, []).append(ttfc)

    if not ttfc_by_input_length:
        return None

    rows = []
    for input_length in sorted(ttfc_by_input_length):
        stats = _compute_stats(ttfc_by_input_length[input_length])
        rows.append({"input_length": input_length, "ttfc": stats})
    return rows


def _report_prefill(cfg: MicrobenchmarkConfig) -> None:
    rows = _collect_prefill_results(cfg)
    if not rows:
        return

    table = Table(title="Prefill Results (TTFC)")
    table.add_column("Input Length", justify="right", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")
    table.add_column("Min (ms)", justify="right", style="dim")
    table.add_column("Max (ms)", justify="right", style="dim")
    table.add_column("Count", justify="right", style="dim")

    for row in rows:
        stats = row["ttfc"]
        table.add_row(
            str(row["input_length"]),
            _fmt_ms(stats["mean"]), _fmt_ms(stats["median"]), _fmt_ms(stats["p99"]),
            _fmt_ms(stats["min"]), _fmt_ms(stats["max"]), str(stats["count"]),
        )

    console.print()
    console.print(table)
    console.print()

    _save_json(
        {"type": "prefill", "metric": "ttfc", "results": rows},
        os.path.join(cfg.output_dir, "prefill_results.json"),
    )


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def _collect_decode_results(cfg: MicrobenchmarkConfig) -> list[dict[str, Any]] | None:
    # Each param combo is in cfg.output_dir/bs=N_il=M/
    all_runs = _find_all_run_metrics(cfg.output_dir)
    if not all_runs:
        return None

    rows: list[tuple[int, int, dict]] = []

    for metrics_dir, records in all_runs:
        batch_size = len(records)
        input_length = records[0]["target_num_delta_prompt_tokens"]

        decode_window_stats = _load_decode_window_stats(metrics_dir)
        if decode_window_stats and decode_window_stats.get("count", 0) > 0:
            stats = decode_window_stats
        else:
            all_time_between_completions = [tbc_value for record in records for tbc_value in record.get("tbc", [])]
            if not all_time_between_completions:
                continue
            stats = _compute_stats(all_time_between_completions)

        rows.append((batch_size, input_length, stats))

    if not rows:
        return None

    rows.sort(key=lambda row: (row[0], row[1]))
    return [
        {"batch_size": batch_size, "input_length": input_length, "tbt": stats}
        for batch_size, input_length, stats in rows
    ]


def _report_decode(cfg: MicrobenchmarkConfig) -> None:
    rows = _collect_decode_results(cfg)
    if not rows:
        return

    table = Table(title="Decode Results (TBT in decode window)")
    table.add_column("Batch Size", justify="right", style="cyan")
    table.add_column("Input Length", justify="right", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")
    table.add_column("Min (ms)", justify="right", style="dim")
    table.add_column("Max (ms)", justify="right", style="dim")
    table.add_column("Samples", justify="right", style="dim")

    for row in rows:
        stats = row["tbt"]
        table.add_row(
            str(row["batch_size"]), str(row["input_length"]),
            _fmt_ms(stats.get("mean")), _fmt_ms(stats.get("median")), _fmt_ms(stats.get("p99")),
            _fmt_ms(stats.get("min")), _fmt_ms(stats.get("max")), str(stats.get("count", "—")),
        )

    console.print()
    console.print(table)
    console.print()

    _save_json(
        {"type": "decode", "metric": "tbt", "results": rows},
        os.path.join(cfg.output_dir, "decode_results.json"),
    )


# ---------------------------------------------------------------------------
# Mixed
# ---------------------------------------------------------------------------


def _parse_mixed_dir_params(metrics_dir: Path) -> dict[str, int] | None:
    """Extract params from a mixed bench dir path like .../bs=4_dil=512_kv=512_dp=256/bench/..."""
    import re
    path_str = str(metrics_dir)
    match = re.search(r"bs=(\d+)_dil=(\d+)_kv=(\d+)_dp=(\d+)", path_str)
    if not match:
        return None
    return {
        "batch_size": int(match.group(1)),
        "decode_input_length": int(match.group(2)),
        "prefill_kv_length": int(match.group(3)),
        "incremental_prefill_size": int(match.group(4)),
    }


def _collect_mixed_results(cfg: MicrobenchmarkConfig) -> list[dict[str, Any]] | None:
    # Each param combo is in cfg.output_dir/bs=N_dil=M_kv=K_dp=D/bench/
    # Skip warmup dirs — only look at bench dirs
    all_runs = _find_all_run_metrics(cfg.output_dir)
    if not all_runs:
        return None

    rows: list[dict[str, Any]] = []

    for metrics_dir, records in all_runs:
        # Skip warmup runs (they're in .../warmup/... paths)
        if "/warmup/" in str(metrics_dir):
            continue

        params = _parse_mixed_dir_params(metrics_dir)

        sorted_by_session = sorted(records, key=lambda record: record["session_id"])
        decode_requests = [record for record in sorted_by_session if record["num_output_tokens"] > 1]
        if not decode_requests:
            continue

        batch_size = params["batch_size"] if params else len(decode_requests)
        decode_input_length = params["decode_input_length"] if params else decode_requests[0]["target_num_delta_prompt_tokens"]
        prefill_kv_length = params["prefill_kv_length"] if params else 0
        incremental_prefill_size = params["incremental_prefill_size"] if params else 0

        decode_window_stats = _load_decode_window_stats(metrics_dir)
        if decode_window_stats and decode_window_stats.get("count", 0) > 0:
            stats = decode_window_stats
        else:
            all_time_between_completions = [tbc_value for record in decode_requests for tbc_value in record.get("tbc", [])]
            if not all_time_between_completions:
                continue
            stats = _compute_stats(all_time_between_completions)

        rows.append({
            "batch_size": batch_size,
            "decode_input_length": decode_input_length,
            "prefill_kv_length": prefill_kv_length,
            "incremental_prefill_size": incremental_prefill_size,
            "tbt": stats,
        })

    if not rows:
        return None

    rows.sort(key=lambda row: (row["batch_size"], row["decode_input_length"], row["prefill_kv_length"], row["incremental_prefill_size"]))
    return rows


def _report_mixed(cfg: MicrobenchmarkConfig) -> None:
    rows = _collect_mixed_results(cfg)
    if not rows:
        return

    table = Table(title="Mixed Batch Results (TBT under interference)")
    table.add_column("Batch Size", justify="right", style="cyan")
    table.add_column("Decode Input Len", justify="right", style="cyan")
    table.add_column("Delta Prefill Len", justify="right", style="cyan")
    table.add_column("Prefill KV Len", justify="right", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")
    table.add_column("Min (ms)", justify="right", style="dim")
    table.add_column("Max (ms)", justify="right", style="dim")
    table.add_column("Samples", justify="right", style="dim")

    for row in rows:
        stats = row["tbt"]
        table.add_row(
            str(row["batch_size"]), str(row["decode_input_length"]),
            str(row["incremental_prefill_size"]),
            str(row["prefill_kv_length"]),
            _fmt_ms(stats.get("mean")), _fmt_ms(stats.get("median")), _fmt_ms(stats.get("p99")),
            _fmt_ms(stats.get("min")), _fmt_ms(stats.get("max")), str(stats.get("count", "—")),
        )

    console.print()
    console.print(table)
    console.print()

    _save_json(
        {"type": "mixed", "metric": "tbt", "results": rows},
        os.path.join(cfg.output_dir, "mixed_results.json"),
    )
