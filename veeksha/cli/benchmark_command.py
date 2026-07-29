"""Nested ``veeksha benchmark`` command group: ``run`` and ``define``."""

from vidhi import BaseCommand, frozen_dataclass


@frozen_dataclass
class BenchmarkCommand(BaseCommand):
    """Named and ad-hoc benchmarks.

    Subcommands:

    * ``veeksha benchmark run`` — run a benchmark (default)
    * ``veeksha benchmark define`` — pin a named-benchmark definition
    """
