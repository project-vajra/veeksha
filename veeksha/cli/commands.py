"""Veeksha top-level CLI command group.

    veeksha benchmark [options]
    veeksha capacity-search [options]
    veeksha prefill [options]
    veeksha decode [options]
    veeksha stress [options]
"""

from __future__ import annotations

import warnings

# Suppress GIL re-enable warnings from C extensions (e.g. tokenizers)
# that haven't declared free-threaded support yet.
warnings.filterwarnings("ignore", message=".*global interpreter lock.*", category=RuntimeWarning)

from vidhi import parse_cli_sweep

from veeksha.capacity_search import run_capacity_search
from veeksha.cli.base import VeekshaCommand
from veeksha.cli.benchmarks import run_cli as run_benchmark
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.microbench.config import (
    DecodeMicrobenchmarkConfig,
    PrefillMicrobenchmarkConfig,
    StressMicrobenchmarkConfig,
)
from veeksha.microbench.decode import run_decode
from veeksha.microbench.prefill import run_prefill
from veeksha.microbench.stress import run_stress

_RUNNERS = {
    BenchmarkConfig: run_benchmark,
    CapacitySearchConfig: lambda configs: [run_capacity_search(c) for c in configs],
    PrefillMicrobenchmarkConfig: lambda configs: [run_prefill(c) for c in configs],
    DecodeMicrobenchmarkConfig: lambda configs: [run_decode(c) for c in configs],
    StressMicrobenchmarkConfig: lambda configs: [run_stress(c) for c in configs],
}


def main() -> None:
    """Entry point for the veeksha CLI."""
    configs = parse_cli_sweep(VeekshaCommand)
    _RUNNERS[type(configs[0])](configs)
