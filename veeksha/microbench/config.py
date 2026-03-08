"""Microbenchmark configuration with inheritance."""

from dataclasses import field
from enum import StrEnum
from typing import ClassVar

from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass


class StressTrafficMode(StrEnum):
    """Traffic pattern for stress microbenchmarks."""

    FIXED_CLIENTS = "fixed-clients"
    FIXED_RATE = "fixed-rate"


@frozen_dataclass
class BaseMicrobenchmarkConfig:
    """Shared fields for all microbenchmark types."""

    model: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Model name"},
    )
    api_base: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "API base URL"},
    )
    api_key: str = field(
        default="dummy",
        metadata={"help": "API key"},
    )
    input_lengths: list[int] = field(
        default_factory=lambda: [128, 256, 512, 1024],
        metadata={"help": "Input lengths for benchmarks"},
    )
    samples_per_length: int = field(
        default=10,
        metadata={"help": "Number of samples per input length"},
    )
    output_dir: str = field(
        default="microbench_output",
        metadata={"help": "Output directory for benchmark results"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"},
    )
    request_timeout: int = field(
        default=120,
        metadata={"help": "Request timeout in seconds"},
    )
    benchmark_timeout: int = field(
        default=600,
        metadata={"help": "Benchmark timeout in seconds"},
    )
    max_tokens_param: str = field(
        default="max_tokens",
        metadata={"help": "Parameter name for max tokens"},
    )
    ignore_eos: bool = field(
        default=True,
        metadata={"help": "Ignore EOS token"},
    )
    validate_only: bool = field(
        default=False,
        metadata={"help": "Skip benchmark, only validate existing output"},
    )
    skip_validation: bool = field(
        default=False,
        metadata={"help": "Skip post-run validation"},
    )


@frozen_dataclass(allow_from_file=True)
class PrefillMicrobenchmarkConfig(BaseMicrobenchmarkConfig):
    """Prefill microbenchmark configuration."""

    output_tokens: int = field(
        default=1,
        metadata={"help": "Output tokens per request"},
    )

    def __post_init__(self) -> None:
        if not self.input_lengths:
            raise ValueError("input_lengths must be non-empty")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        if self.samples_per_length <= 0:
            raise ValueError("samples_per_length must be positive")

    @classmethod
    def create_from_cli_args(cls) -> list["PrefillMicrobenchmarkConfig"]:
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        return [fc.reconstruct_original_dataclass() for fc in flat_configs]


@frozen_dataclass(allow_from_file=True)
class DecodeMicrobenchmarkConfig(BaseMicrobenchmarkConfig):
    """Decode microbenchmark configuration."""

    batch_sizes: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8],
        metadata={"help": "Batch sizes for decode benchmarks"},
    )
    engine_chunk_size: int = field(
        default=512,
        metadata={"help": "Engine chunk size"},
    )

    def __post_init__(self) -> None:
        if not self.input_lengths:
            raise ValueError("input_lengths must be non-empty")
        if not self.batch_sizes:
            raise ValueError("batch_sizes must be non-empty")
        if self.samples_per_length <= 0:
            raise ValueError("samples_per_length must be positive")
        if self.engine_chunk_size <= 0:
            raise ValueError("engine_chunk_size must be positive")
        for bs in self.batch_sizes:
            if bs >= self.engine_chunk_size:
                raise ValueError(
                    f"batch_size {bs} must be less than engine_chunk_size {self.engine_chunk_size}"
                )

    @classmethod
    def create_from_cli_args(cls) -> list["DecodeMicrobenchmarkConfig"]:
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        return [fc.reconstruct_original_dataclass() for fc in flat_configs]


# ---------------------------------------------------------------------------
# Stress microbenchmark configs (polymorphic by mode)
# ---------------------------------------------------------------------------


def _validate_stress_base(cfg: "StressMicrobenchmarkConfig") -> None:
    """Shared validation for all stress config variants."""
    if cfg.input_length <= 0:
        raise ValueError("input_length must be positive")
    if cfg.output_length <= 0:
        raise ValueError("output_length must be positive")
    if cfg.point_duration <= cfg.warmup_duration:
        raise ValueError("point_duration must exceed warmup_duration")
    if not isinstance(cfg.traffic_mode, StressTrafficMode):
        try:
            StressTrafficMode(cfg.traffic_mode)
        except ValueError:
            raise ValueError(
                f"traffic_mode must be one of {[m.value for m in StressTrafficMode]}, "
                f"got {cfg.traffic_mode!r}"
            )


@frozen_dataclass
class StressMicrobenchmarkConfig(BaseMicrobenchmarkConfig):
    """Base config for stress microbenchmarks (throughput-vs-latency curves)."""

    STRESS_MODE: ClassVar[str]  # set by each subclass

    input_length: int = field(
        default=512,
        metadata={"help": "Input token length (single value)"},
    )
    output_length: int = field(
        default=128,
        metadata={"help": "Output token length (single value)"},
    )
    point_duration: int = field(
        default=120,
        metadata={"help": "Seconds to run each concurrency point"},
    )
    warmup_duration: int = field(
        default=10,
        metadata={"help": "Warmup seconds to discard per point"},
    )
    traffic_mode: StressTrafficMode = field(
        default=StressTrafficMode.FIXED_CLIENTS,
        metadata={
            "help": "Traffic pattern: 'fixed-clients' (N clients sending back-to-back) "
            "or 'fixed-rate' (Poisson arrivals at N req/s)"
        },
    )
    max_tokens_per_second_estimate: float = field(
        default=500.0,
        metadata={"help": "Estimated max output tok/s (for session budget)"},
    )

    @classmethod
    def create_from_cli_args(cls) -> list["StressMicrobenchmarkConfig"]:
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        return [fc.reconstruct_original_dataclass() for fc in flat_configs]


@frozen_dataclass(allow_from_file=True)
class ManualStressConfig(StressMicrobenchmarkConfig):
    """Stress config with explicit concurrency levels."""

    STRESS_MODE: ClassVar[str] = "manual"

    concurrency_levels: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32],
        metadata={"help": "Concurrency levels to test"},
    )

    def __post_init__(self) -> None:
        _validate_stress_base(self)
        if not self.concurrency_levels:
            raise ValueError("concurrency_levels must be non-empty")
        if any(c <= 0 for c in self.concurrency_levels):
            raise ValueError("all concurrency_levels must be positive")

    @classmethod
    def create_from_cli_args(cls) -> list["StressMicrobenchmarkConfig"]:
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        return [fc.reconstruct_original_dataclass() for fc in flat_configs]


@frozen_dataclass(allow_from_file=True)
class RangeStressConfig(StressMicrobenchmarkConfig):
    """Stress config with log-spaced concurrency range."""

    STRESS_MODE: ClassVar[str] = "range"

    concurrency_min: int = field(
        default=1,
        metadata={"help": "Minimum concurrency level"},
    )
    concurrency_max: int = field(
        default=64,
        metadata={"help": "Maximum concurrency level"},
    )
    concurrency_points: int = field(
        default=8,
        metadata={"help": "Number of log-spaced points to test"},
    )

    def __post_init__(self) -> None:
        _validate_stress_base(self)
        if self.concurrency_min >= self.concurrency_max:
            raise ValueError("concurrency_min must be less than concurrency_max")
        if self.concurrency_points <= 0:
            raise ValueError("concurrency_points must be positive")

    @classmethod
    def create_from_cli_args(cls) -> list["StressMicrobenchmarkConfig"]:
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        return [fc.reconstruct_original_dataclass() for fc in flat_configs]


@frozen_dataclass(allow_from_file=True)
class AutoStressConfig(StressMicrobenchmarkConfig):
    """Stress config with automatic concurrency discovery."""

    STRESS_MODE: ClassVar[str] = "auto"

    auto_throughput_threshold: float = field(
        default=0.05,
        metadata={"help": "Stop probing when throughput gain < this fraction"},
    )
    auto_max_probes: int = field(
        default=20,
        metadata={"help": "Maximum number of probe points"},
    )
    auto_fill_points: int = field(
        default=8,
        metadata={"help": "Number of fill points between lower and upper bounds"},
    )
    resume_dir: str = field(
        default="",
        metadata={
            "help": "Resume from a previous run directory (reuse existing c=N results)"
        },
    )

    def __post_init__(self) -> None:
        _validate_stress_base(self)
        if self.auto_max_probes <= 0:
            raise ValueError("auto_max_probes must be positive")
        if self.auto_fill_points <= 0:
            raise ValueError("auto_fill_points must be positive")

    @classmethod
    def create_from_cli_args(cls) -> list["StressMicrobenchmarkConfig"]:
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        return [fc.reconstruct_original_dataclass() for fc in flat_configs]


STRESS_MODE_TO_CONFIG: dict[str, type["StressMicrobenchmarkConfig"]] = {
    "manual": ManualStressConfig,
    "range": RangeStressConfig,
    "auto": AutoStressConfig,
}
