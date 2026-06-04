"""Base command class for Veeksha CLI subcommands."""

from vidhi import BaseCommand, frozen_dataclass


@frozen_dataclass
class VeekshaCommand(BaseCommand):
    """Veeksha - high-fidelity benchmarking for LLM inference systems."""
