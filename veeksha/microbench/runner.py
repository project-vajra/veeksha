"""CLI runner for veeksha microbenchmarks."""

import os
import sys
from dataclasses import replace
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from veeksha.cli.benchmarks import BenchmarkCliRunner
from veeksha.logger import init_logger
from veeksha.microbench.config import MicrobenchmarkConfig
from veeksha.microbench.config_builder import build_benchmark_configs
from veeksha.microbench.results import print_results_table
from veeksha.microbench.validate import validate

logger = init_logger(__name__)
console = Console()

# Consistent timestamp format across microbench: YYYY-MM-DD_HH-MM-SS
_TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"


def _make_run_dir(cfg: MicrobenchmarkConfig) -> MicrobenchmarkConfig:
    """Create a unique timestamped directory for this microbenchmark invocation."""
    timestamp = datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT)
    run_dir = os.path.join(cfg.output_dir, f"{cfg.type}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return replace(cfg, output_dir=run_dir)


def _print_banner(cfg: MicrobenchmarkConfig) -> None:
    """Print a startup banner summarizing the microbenchmark configuration."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Model", cfg.model)
    table.add_row("API base", cfg.api_base)
    table.add_row("Output dir", cfg.output_dir)

    if cfg.type == "prefill":
        table.add_row("Input lengths", str(cfg.input_lengths))
        table.add_row("Output tokens", str(cfg.output_tokens))
        table.add_row("Samples/length", str(cfg.samples_per_length))
    elif cfg.type == "decode":
        table.add_row("Batch sizes", str(cfg.batch_sizes))
        table.add_row("Input lengths", str(cfg.input_lengths))
        table.add_row("Samples/length", str(cfg.samples_per_length))
        table.add_row("Chunk size", str(cfg.engine_chunk_size))
    elif cfg.type == "mixed":
        table.add_row("Batch sizes", str(cfg.batch_sizes))
        table.add_row("Decode input lengths", str(cfg.decode_input_lengths))
        table.add_row("Prefill KV lengths", str(cfg.prefill_kv_lengths))
        table.add_row("Incr. prefill sizes", str(cfg.incremental_prefill_sizes))
        table.add_row("Samples/length", str(cfg.samples_per_length))
        table.add_row("Chunk size", str(cfg.engine_chunk_size))

    console.print()
    console.print(
        Panel(
            table,
            title=f"[bold]Veeksha Microbenchmark — {cfg.type}[/bold]",
            border_style="blue",
        )
    )
    console.print()


def _print_validation_failure(result) -> None:
    """Print validation results only when there are failures."""
    table = Table(title="Post-Run Validation — Failures Detected", border_style="red")
    table.add_column("Status", justify="center", width=6)
    table.add_column("Check")
    table.add_column("Detail", style="dim")

    for status, name, detail in result.checks:
        if status == "PASS":
            style_tag = "[green]PASS[/green]"
        elif status == "WARN":
            style_tag = "[yellow]WARN[/yellow]"
        else:
            style_tag = "[red bold]FAIL[/red bold]"
        table.add_row(style_tag, name, detail)

    num_passed = sum(1 for status, _, _ in result.checks if status == "PASS")
    num_warnings = sum(1 for status, _, _ in result.checks if status == "WARN")
    num_failures = sum(1 for status, _, _ in result.checks if status == "FAIL")

    console.print()
    console.print(table)
    console.print(
        f"  [green]{num_passed} passed[/green], [yellow]{num_warnings} warnings[/yellow], [red]{num_failures} failures[/red]"
    )
    console.print()


def main() -> None:
    configs = MicrobenchmarkConfig.create_from_cli_args()
    for cfg in configs:
        cfg = _make_run_dir(cfg)
        _print_banner(cfg)

        if not cfg.validate_only:
            benchmark_configs = build_benchmark_configs(cfg)
            BenchmarkCliRunner(benchmark_configs).run_all()

        print_results_table(cfg)

        if not cfg.skip_validation:
            result = validate(cfg, cfg.output_dir)
            if result.ok:
                num_passed = sum(
                    1 for status, _, _ in result.checks if status == "PASS"
                )
                logger.info(f"Validation passed ({num_passed} checks)")
            else:
                _print_validation_failure(result)
                sys.exit(1)
