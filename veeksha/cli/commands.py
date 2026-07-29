"""Veeksha top-level CLI command group.

veeksha benchmark run [options]
veeksha benchmark define [options]
veeksha capacity-search [options]
veeksha prefill | decode | stress | diff | score-tts-longform | health [options]

Ad-hoc one-shot runs (no named definition) use top-level flags and dispatch to
``benchmark run``::

    veeksha --endpoint.api_base http://localhost:8000/v1 --endpoint.model m
"""

from __future__ import annotations

import sys
import warnings

# Suppress GIL re-enable warnings from C extensions (e.g. tokenizers)
# that haven't declared free-threaded support yet.
warnings.filterwarnings(
    "ignore", message=".*global interpreter lock.*", category=RuntimeWarning
)

# Import so BenchmarkCommand registers run/define subcommands.
import veeksha.config.benchmark  # noqa: F401
import veeksha.config.benchmark_define  # noqa: F401
from veeksha.benchmark_define import run_benchmark_define_cli
from veeksha.capacity_search import run_capacity_search
from veeksha.cli.base import VeekshaCommand
from veeksha.cli.benchmark_command import BenchmarkCommand
from veeksha.cli.benchmark_run_cli import parse_benchmark_run_configs
from veeksha.cli.benchmarks import run_cli as run_benchmark
from veeksha.cli.parsing import parse_cli_sweep
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.benchmark_define import BenchmarkDefineConfig
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.config.health_check import HealthCheckConfig
from veeksha.config.score_tts_longform import ScoreTtsLongformConfig
from veeksha.health import run_health_check_cli
from veeksha.microbench.config import (
    DecodeMicrobenchmarkConfig,
    PrefillMicrobenchmarkConfig,
    StressMicrobenchmarkConfig,
)
from veeksha.microbench.decode import run_decode
from veeksha.microbench.diff import DiffConfig, run_diff
from veeksha.microbench.prefill import run_prefill
from veeksha.microbench.stress import run_stress
from veeksha.verification.longform import run_score_tts_longform_cli
from veeksha.version import __version__

_RUNNERS = {
    BenchmarkConfig: run_benchmark,
    BenchmarkDefineConfig: run_benchmark_define_cli,
    CapacitySearchConfig: lambda configs: [run_capacity_search(c) for c in configs],
    PrefillMicrobenchmarkConfig: lambda configs: [run_prefill(c) for c in configs],
    DecodeMicrobenchmarkConfig: lambda configs: [run_decode(c) for c in configs],
    StressMicrobenchmarkConfig: lambda configs: [run_stress(c) for c in configs],
    DiffConfig: lambda configs: [run_diff(c) for c in configs],
    ScoreTtsLongformConfig: run_score_tts_longform_cli,
    HealthCheckConfig: run_health_check_cli,
}

_VERSION_FLAGS = {"--version", "-V"}

_TOP_LEVEL_OTHER = frozenset(
    {
        "capacity-search",
        "prefill",
        "decode",
        "stress",
        "diff",
        "score-tts-longform",
        "health",
    }
)


def _dispatch(configs: list) -> None:
    if not configs:
        sys.exit("No configuration resolved from CLI arguments.")
    runner = _RUNNERS.get(type(configs[0]))
    if runner is None:
        sys.exit(f"Unknown command config type: {type(configs[0]).__name__}")
    runner(configs)


def _dispatch_benchmark_group(args: list[str]) -> None:
    """Handle ``veeksha benchmark [run|define] ...``."""
    if not args or args[0] in ("-h", "--help"):
        # Group help (lists run + define).
        _dispatch(parse_cli_sweep(BenchmarkCommand, args=args or ["-h"]))
        return

    if args[0] == "define":
        _dispatch(parse_cli_sweep(BenchmarkDefineConfig, args=args[1:]))
        return

    if args[0] == "run":
        _dispatch(parse_benchmark_run_configs(args[1:]))
        return

    # Flags after ``benchmark`` with no subcommand → run (default).
    if args[0].startswith("-"):
        _dispatch(parse_benchmark_run_configs(args))
        return

    available = "run, define"
    sys.exit(f"Error: Unknown benchmark command '{args[0]}'. Available: {available}")


def main() -> None:
    """Entry point for the veeksha CLI."""
    if len(sys.argv) == 2 and sys.argv[1] in _VERSION_FLAGS:
        print(f"veeksha {__version__}")
        return

    argv = sys.argv[1:]

    # Nested group: veeksha benchmark run|define ...
    if argv[:1] == ["benchmark"]:
        _dispatch_benchmark_group(argv[1:])
        return

    # Ad-hoc one-shot: veeksha --endpoint.api_base ... → benchmark run
    if not argv or argv[0].startswith("-"):
        _dispatch(parse_benchmark_run_configs(argv))
        return

    # Other top-level commands.
    if argv[0] in _TOP_LEVEL_OTHER or argv[0] in VeekshaCommand._subcommands:
        _dispatch(parse_cli_sweep(VeekshaCommand, args=argv))
        return

    available = ", ".join(
        sorted({"benchmark", *_TOP_LEVEL_OTHER, *VeekshaCommand._subcommands.keys()})
    )
    sys.exit(f"Error: Unknown command '{argv[0]}'. Available: {available}")
