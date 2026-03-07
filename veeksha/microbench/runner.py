"""Shared run logic for veeksha microbenchmarks."""

import os
import sys
from dataclasses import replace
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from veeksha.cli.benchmarks import BenchmarkCliRunner
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.logger import init_logger
from veeksha.microbench.common import ValidationResult
from veeksha.microbench.config import BaseMicrobenchmarkConfig

logger = init_logger(__name__)
console = Console()

_TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"


def run(
    cfg: BaseMicrobenchmarkConfig,
    type_name: str,
    banner_rows: list[tuple[str, str]],
    build_benchmark_configs,
    print_results_table,
    validate,
) -> None:
    """Run a single microbenchmark: banner → build → execute → report → validate."""
    cfg = _make_run_dir(cfg, type_name)
    _print_banner(cfg, type_name, banner_rows)

    if not cfg.validate_only:
        benchmark_configs: list[BenchmarkConfig] = build_benchmark_configs(cfg)
        BenchmarkCliRunner(benchmark_configs).run_all()

    print_results_table(cfg)

    if not cfg.skip_validation:
        result: ValidationResult = validate(cfg, cfg.output_dir)
        if result.ok:
            num_passed = sum(1 for s, _, _ in result.checks if s == "PASS")
            logger.info(f"Validation passed ({num_passed} checks)")
        else:
            _print_validation_failure(result)
            sys.exit(1)


def _make_run_dir(cfg: BaseMicrobenchmarkConfig, type_name: str) -> BaseMicrobenchmarkConfig:
    """Create a unique timestamped directory for this microbenchmark invocation."""
    timestamp = datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT)
    run_dir = os.path.join(cfg.output_dir, f"{type_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return replace(cfg, output_dir=run_dir)


def _print_banner(
    cfg: BaseMicrobenchmarkConfig,
    type_name: str,
    banner_rows: list[tuple[str, str]],
) -> None:
    """Print a startup banner summarizing the microbenchmark configuration."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Model", cfg.model)
    table.add_row("API base", cfg.api_base)
    table.add_row("Output dir", cfg.output_dir)

    for label, attr in banner_rows:
        table.add_row(label, str(getattr(cfg, attr)))

    console.print()
    console.print(
        Panel(
            table,
            title=f"[bold]Veeksha Microbenchmark — {type_name}[/bold]",
            border_style="blue",
        )
    )
    console.print()


def _print_validation_failure(result: ValidationResult) -> None:
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

    num_passed = sum(1 for s, _, _ in result.checks if s == "PASS")
    num_warnings = sum(1 for s, _, _ in result.checks if s == "WARN")
    num_failures = sum(1 for s, _, _ in result.checks if s == "FAIL")

    console.print()
    console.print(table)
    console.print(
        f"  [green]{num_passed} passed[/green], [yellow]{num_warnings} warnings[/yellow], [red]{num_failures} failures[/red]"
    )
    console.print()
