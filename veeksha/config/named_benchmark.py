"""Configuration for running a versioned named benchmark."""

from pathlib import Path

from vidhi import field, frozen_dataclass

from veeksha.cli.base import VeekshaCommand


@frozen_dataclass
class NamedBenchmarkConfig(VeekshaCommand, name="named-benchmark"):
    """Compile and run a catalog benchmark against one target configuration."""

    benchmark: str = field(
        "",
        help=(
            "Catalog benchmark ID (for example "
            "asr.indic.multidomain16.v1) or a path to a benchmark manifest "
            "YAML file."
        ),
    )
    target_config: str = field(
        "",
        help=(
            "Path to an ordinary Veeksha BenchmarkConfig YAML. Only its client, "
            "endpoint, and server binding are used; the named benchmark owns the "
            "workload, metrics, and runtime contract. !expand model sweeps are "
            "supported."
        ),
    )
    output_dir: str = field(
        "benchmark_output/named",
        help="Parent directory for the named benchmark and its dataset runs.",
    )
    dataset_root: str = field(
        "",
        help=(
            "Optional local dataset root used to resolve ${DATASET_ROOT} in "
            "catalog session-generator paths."
        ),
    )
    dry_run: bool = field(
        False,
        help=(
            "Materialize and validate every dataset/target BenchmarkConfig without "
            "calling any endpoint."
        ),
    )

    def __post_init__(self) -> None:
        if not self.benchmark.strip():
            raise ValueError("NamedBenchmarkConfig.benchmark is required.")
        if not self.target_config.strip():
            raise ValueError("NamedBenchmarkConfig.target_config is required.")
        if not Path(self.target_config).is_file():
            raise ValueError(
                f"Named benchmark target_config not found: {self.target_config}"
            )
        if not self.output_dir.strip():
            raise ValueError("NamedBenchmarkConfig.output_dir is required.")
