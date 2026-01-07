"""Global pytest configuration and fixtures."""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, cast

import pytest
from pytest import Config, Item, TestReport
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.markup import escape
from rich.table import Table
from rich.theme import Theme

# heavy deps for functional/GPU tests
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

# Regex to extract parameters from a test nodeid
PARAM_PATTERN = re.compile(r"\[(.*?)\]$")


@pytest.fixture
def gpu_test_sync_cuda():
    """Synchronize CUDA before and after GPU tests."""
    if _TORCH_AVAILABLE and getattr(torch, "cuda", None) and torch.cuda.is_available():  # type: ignore[attr-defined]
        torch.cuda.synchronize()  # type: ignore[union-attr]
    yield
    if _TORCH_AVAILABLE and getattr(torch, "cuda", None) and torch.cuda.is_available():  # type: ignore[attr-defined]
        torch.cuda.synchronize()  # type: ignore[union-attr]


@dataclass
class TestRunStats:
    """Tracks statistics for test execution."""

    start_time: float = 0.0
    total_duration: float = 0.0
    tests_executed: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    tests_deselected: int = 0
    slowest_tests: List[Tuple[str, float]] = field(default_factory=list)

    def __post_init__(self):
        self.slowest_tests = []
        self.start_time = time.time()

    def update_duration(self):
        self.total_duration = time.time() - self.start_time

    def record_test_duration(self, test_name: str, duration: float):
        self.slowest_tests.append((test_name, duration))
        self.slowest_tests.sort(key=lambda x: x[1], reverse=True)
        if len(self.slowest_tests) > 10:
            self.slowest_tests.pop()


def is_running_in_ci() -> bool:
    """Detect if the code is running in a CI environment."""
    ci_env_vars = [
        "CI",
        "CONTINUOUS_INTEGRATION",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "JENKINS_URL",
        "TRAVIS",
        "CIRCLECI",
    ]
    return any(os.environ.get(var) for var in ci_env_vars)


custom_theme = Theme(
    {
        "success": "green",
        "failure": "red",
        "skip": "yellow",
        "warning": "yellow",
        "info": "cyan",
        "debug": "blue",
        "title": "magenta bold",
        "unit": "cyan",
        "functional": "blue",
        "gpu": "magenta",
    }
)

console = Console(theme=custom_theme, force_terminal=True)
stats = TestRunStats()
_pytest_config: Optional[Config] = None
_current_progress = None
_test_progress_task = None
_use_progress_bar: bool = not is_running_in_ci()
_collected_items_count: int = 0
_first_test_seen: bool = False


def pytest_configure(config: Config) -> None:
    """Pytest hook to capture config object and initialize progress bar."""
    global _pytest_config, _current_progress, _test_progress_task, _use_progress_bar
    _pytest_config = config

    # Check for command-line option to override progress bar behavior
    if hasattr(config.option, "no_progress_bar") and config.option.no_progress_bar:
        _use_progress_bar = False

    # Initialize progress bar if not in CI environment
    if _use_progress_bar:
        _current_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
            TextColumn("[green]{task.fields[passed]} passed"),
            TextColumn("[red]{task.fields[failed]} failed"),
            TextColumn("[yellow]{task.fields[skipped]} skipped"),
            TextColumn("[cyan]{task.elapsed:.2f}s"),
            console=console,
            expand=True,
        )
        _current_progress.start()


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--no-progress-bar",
        action="store_true",
        default=False,
        help="Disable progress bar regardless of environment",
    )


def pytest_collection_modifyitems(config: Config, items: List[Item]) -> None:
    """Pytest hook to initialize progress bar and auto-add GPU fixtures."""
    global _test_progress_task, _current_progress, _collected_items_count

    _collected_items_count = len(items)

    if _current_progress and _use_progress_bar:
        _test_progress_task = _current_progress.add_task(
            "Running tests", total=_collected_items_count, passed=0, failed=0, skipped=0
        )

    # Auto-add GPU sync fixture to GPU tests
    for item in items:
        if "gpu" in item.keywords:
            item_any = cast(Any, item)
            if "gpu_test_sync_cuda" not in getattr(item_any, "fixturenames", []):
                item_any.fixturenames.append("gpu_test_sync_cuda")


def pytest_deselected(items):
    """Pytest hook for tracking deselected items."""
    global stats
    stats.tests_deselected += len(items)


def extract_parameters(nodeid: str) -> str:
    """Extract parameter info from test nodeid."""
    match = PARAM_PATTERN.search(nodeid)
    if match:
        return match.group(1)
    return ""


def test_category_style(nodeid: str) -> str:
    """Determine style based on test category."""
    if "unit" in nodeid:
        return "unit"
    elif "functional" in nodeid:
        return "functional"
    elif "gpu" in nodeid:
        return "gpu"
    else:
        return ""


def get_formatted_test_name(report: TestReport) -> str:
    """Format test nodeid with parameters and category styling."""
    params = extract_parameters(report.nodeid)
    category = test_category_style(report.nodeid)

    if params:
        if category:
            return f"[{category}]{escape(report.nodeid.split('[')[0])}[/{category}] [bold]({escape(params)})[/bold]"
        else:
            return f"{escape(report.nodeid.split('[')[0])} [bold]({escape(params)})[/bold]"
    else:
        if category:
            return f"[{category}]{escape(report.nodeid)}[/{category}]"
        else:
            return f"{escape(report.nodeid)}"


def update_progress(outcome: str) -> None:
    """Update the progress bar counters."""
    global _current_progress, _test_progress_task, _first_test_seen, stats

    if not _use_progress_bar or not _current_progress or _test_progress_task is None:
        return

    # On first test, adjust progress bar total to account for deselected tests
    if not _first_test_seen:
        _first_test_seen = True
        if stats.tests_deselected > 0:
            new_total = _collected_items_count - stats.tests_deselected
            _current_progress.update(_test_progress_task, total=new_total)

    # Update outcome counters
    if outcome == "passed":
        current_passed = _current_progress.tasks[_test_progress_task].fields["passed"]
        _current_progress.update(_test_progress_task, advance=1, passed=current_passed + 1)
    elif outcome == "failed":
        current_failed = _current_progress.tasks[_test_progress_task].fields["failed"]
        _current_progress.update(_test_progress_task, advance=1, failed=current_failed + 1)
    elif outcome == "skipped":
        current_skipped = _current_progress.tasks[_test_progress_task].fields["skipped"]
        _current_progress.update(_test_progress_task, advance=1, skipped=current_skipped + 1)


def pytest_runtest_logreport(report: TestReport) -> None:
    """Hook to process test reports and update progress bar and statistics."""
    if not _pytest_config:
        raise RuntimeError("pytest_configure was not called")

    # Handle test outcomes during setup phase
    if report.when == "setup":
        if report.outcome == "skipped":
            stats.tests_skipped += 1
            stats.tests_executed += 1
            update_progress("skipped")
            skip_reason = (
                report.longrepr[2]
                if hasattr(report, "longrepr") and isinstance(report.longrepr, tuple)
                else ""
            )
            console.print(f"[skip]s {get_formatted_test_name(report)} {skip_reason}[/skip]")
        elif report.outcome == "failed":
            stats.tests_failed += 1
            stats.tests_executed += 1
            update_progress("failed")
            console.print(f"[failure]✘ {get_formatted_test_name(report)} (setup failed)[/failure]")

    # Handle test outcomes during call phase
    elif report.when == "call":
        stats.tests_executed += 1
        if report.outcome == "passed":
            stats.tests_passed += 1
            update_progress("passed")
            console.print(f"[success]✔ {get_formatted_test_name(report)}[/success]")
        elif report.outcome == "failed":
            stats.tests_failed += 1
            update_progress("failed")
            console.print(f"[failure]✘ {get_formatted_test_name(report)}[/failure]")
        elif report.outcome == "skipped":
            stats.tests_skipped += 1
            update_progress("skipped")
            skip_reason = (
                report.longrepr[2]
                if hasattr(report, "longrepr") and isinstance(report.longrepr, tuple)
                else ""
            )
            console.print(f"[skip]s {get_formatted_test_name(report)} {skip_reason}[/skip]")

    # Record test duration for statistics
    if report.when == "call" and hasattr(report, "duration"):
        stats.record_test_duration(get_formatted_test_name(report), report.duration)


def pytest_terminal_summary(terminalreporter) -> None:
    """Pytest hook to add customized terminal summary report."""
    global _current_progress

    # Stop the progress bar
    if _use_progress_bar and _current_progress:
        _current_progress.stop()

    stats.update_duration()

    console.rule("[title]Test Summary[/title]")

    # Create summary table
    table = Table(title="Test Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Total tests", str(stats.tests_executed))
    table.add_row("Passed", f"[success]{stats.tests_passed}[/success]")
    table.add_row("Failed", f"[failure]{stats.tests_failed}[/failure]")
    table.add_row("Skipped", f"[skip]{stats.tests_skipped}[/skip]")
    if stats.tests_deselected > 0:
        table.add_row("Deselected", f"{stats.tests_deselected}")
    table.add_row("Total duration", f"{stats.total_duration:.2f}s")

    console.print(table)

    # Show slowest tests
    if stats.slowest_tests:
        slow_table = Table(title="Slowest Tests")
        slow_table.add_column("Test", style="bold")
        slow_table.add_column("Duration (s)")

        for test, duration in stats.slowest_tests[:5]:
            slow_table.add_row(test, f"{duration:.2f}")

        console.print(slow_table)

    console.rule("[title]End Test Summary[/title]")