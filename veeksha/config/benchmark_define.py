"""CLI config for ``veeksha define``."""

from vidhi import field, frozen_dataclass

from veeksha.cli.base import VeekshaCommand


@frozen_dataclass
class BenchmarkDefineConfig(VeekshaCommand, name="define"):
    """Validate a named-benchmark definition, pin its workload, optionally publish.

    Freezes the full config at definition time, computes one workload
    fingerprint from ``config.runtime.max_sessions`` (a normal frozen field,
    not a define flag), and verifies that each declared free variable
    (``knobs``) leaves that fingerprint unchanged. With ``--publish``,
    uploads the self-contained tree to the Hub.

    Invoked as ``veeksha define``.
    """

    definition: str = field(
        "",
        aliases=["def"],
        help=(
            "Path to a benchmark definition directory or its benchmark.yml. "
            "The directory must contain the pinned config and any assets. "
            "Session count for the pin comes from config.runtime.max_sessions."
        ),
    )
    publish: bool = field(
        False,
        help="Upload the pinned definition to the Hugging Face Hub.",
    )
    repo: str = field(
        "",
        help=(
            "Hub dataset repo id. Defaults to $VEEKSHA_BENCHMARKS_REPO or "
            "avartha/veeksha-benchmarks."
        ),
    )
    tag: str = field(
        "",
        help="Optional Hub tag to create after a successful publish.",
    )
    private: bool = field(
        False,
        help="Create the Hub repo as private when publishing.",
    )
    commit_message: str = field(
        "",
        help="Commit message used when publishing. Empty uses a default.",
    )
    output: str = field(
        "",
        help=(
            "Optional directory to write the pinned definition tree into. "
            "Defaults to updating the definition directory in place."
        ),
    )

    def __post_init__(self) -> None:
        if not self.definition:
            raise ValueError("benchmark define requires --definition")
