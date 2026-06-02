"""Command line entrypoint for the orchestrated Veeksha launcher."""

from __future__ import annotations

import argparse
import logging
import resource
import sys

from veeksha_launcher.config import LauncherConfig, LauncherConfigError
from veeksha_launcher.orchestrator import LauncherOrchestrator

logger = logging.getLogger(__name__)


def _raise_open_file_limit() -> None:
    """Raise the soft open-file limit to the hard limit for the sweep.

    High-concurrency sweeps hold one streaming socket plus per-thread asyncio
    loop fds for every client thread, which exhausts the default soft limit
    (often 1024) well before 256 concurrency and surfaces as
    ``OSError: [Errno 24] Too many open files``. Raising it here, before the
    orchestrator starts, means the engine and benchmark child processes inherit
    the higher limit too. Best effort: a failure is logged, not fatal.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read RLIMIT_NOFILE: %s", exc)
        return
    desired = 1 << 20  # 1048576
    if hard != resource.RLIM_INFINITY:
        desired = min(desired, hard)
    if desired <= soft:
        return

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
        logger.info("Raised RLIMIT_NOFILE soft limit from %s to %s", soft, desired)
    except (OSError, ValueError) as exc:
        logger.warning("Could not raise RLIMIT_NOFILE to %s: %s", desired, exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch an engine and run an orchestrated Veeksha sweep."
    )
    parser.add_argument(
        "--config", required=True, help="Path to veeksha_launcher YAML config."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _raise_open_file_limit()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = LauncherConfig.from_file(args.config)
        LauncherOrchestrator(config).run()
    except LauncherConfigError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
