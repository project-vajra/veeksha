"""Configuration for the standalone `veeksha health` command."""

from vidhi import field, frozen_dataclass

from veeksha.cli.base import VeekshaCommand


@frozen_dataclass
class HealthCheckConfig(VeekshaCommand, name="health"):
    """Run all post-run health checks on a finished benchmark output directory.

    Reconstructs the benchmark configuration from the run's config.yml,
    re-runs every health check the benchmark runs inline, and rewrites
    health_check_results.txt. Checks that need live server state (the TTS
    zombie-session probe) cannot be re-run post-hoc; their previously
    recorded sections are carried over from the in-run report.
    """

    run_dir: str = field(
        "",
        aliases=["run-dir"],
        help=(
            "Path to a veeksha benchmark output directory (the timestamped "
            "directory containing config.yml and metrics/)."
        ),
    )
    output_file: str = field(
        "",
        aliases=["output-file"],
        help=(
            "Where to write the health report. Defaults to "
            "<run_dir>/health_check_results.txt."
        ),
    )
    strict: bool = field(
        True,
        help="Exit non-zero when any health check fails.",
    )

    def __post_init__(self) -> None:
        if not self.run_dir:
            raise ValueError("health requires --run_dir")
