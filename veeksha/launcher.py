"""Public CLI/API boundary for orchestrated Veeksha launcher sweeps.

Configuration lives in ``veeksha.config.launcher`` and lifecycle orchestration
lives in ``veeksha.orchestration``. This module is intentionally thin, matching
the public-module shape used by ``veeksha.capacity_search``.
"""

from __future__ import annotations

import argparse
import logging
import resource
import sys
from pathlib import Path

from veeksha.config.launcher import LauncherConfig, LauncherConfigError, RetryConfig
from veeksha.orchestration.launcher import BenchmarkAttemptResult, LauncherOrchestrator

logger = logging.getLogger(__name__)


def _raise_open_file_limit() -> None:
    """Raise the soft open-file limit for high-concurrency launcher sweeps."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read RLIMIT_NOFILE: %s", exc)
        return

    desired = 1 << 20
    if hard != resource.RLIM_INFINITY:
        desired = min(desired, hard)
    if desired <= soft:
        return

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
        logger.info("Raised RLIMIT_NOFILE soft limit from %s to %s", soft, desired)
    except (OSError, ValueError) as exc:
        logger.warning("Could not raise RLIMIT_NOFILE to %s: %s", desired, exc)


def run_launcher(config: LauncherConfig) -> None:
    """Run an already-parsed launcher config."""
    LauncherOrchestrator(config).run()


def run_launcher_file(path: str | Path) -> None:
    """Load a launcher YAML file and run it."""
    run_launcher(LauncherConfig.from_file(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veeksha launcher",
        description="Launch an engine and run an orchestrated Veeksha sweep.",
    )
    parser.add_argument(
        "--config", required=True, help="Path to Veeksha launcher YAML config."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _raise_open_file_limit()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_launcher_file(args.config)
    except LauncherConfigError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        return 130
    return 0


__all__ = [
    "BenchmarkAttemptResult",
    "LauncherConfig",
    "LauncherConfigError",
    "LauncherOrchestrator",
    "RetryConfig",
    "build_parser",
    "main",
    "run_launcher",
    "run_launcher_file",
]


if __name__ == "__main__":
    sys.exit(main())
