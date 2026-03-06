"""Unified microbenchmark configuration."""

from dataclasses import field

from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class MicrobenchmarkConfig:
    """Single config for all microbenchmark types: prefill and decode."""

    type: str = field(
        default="prefill",
        metadata={"help": "Benchmark type: prefill or decode"},
    )
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
        metadata={"help": "Input lengths for prefill/decode benchmarks"},
    )
    output_tokens: int = field(
        default=1,
        metadata={"help": "Output tokens per request (prefill only)"},
    )
    samples_per_length: int = field(
        default=10,
        metadata={"help": "Number of samples per input length"},
    )
    batch_sizes: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8],
        metadata={"help": "Batch sizes for decode benchmarks"},
    )
    engine_chunk_size: int = field(
        default=512,
        metadata={"help": "Engine chunk size (decode)"},
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

    def __post_init__(self) -> None:
        valid_types = ("prefill", "decode")
        if self.type not in valid_types:
            raise ValueError(
                f"Unknown microbenchmark type '{self.type}'. Valid types: {', '.join(sorted(valid_types))}"
            )

        if self.type == "prefill":
            if not self.input_lengths:
                raise ValueError("input_lengths must be non-empty")
            if self.output_tokens <= 0:
                raise ValueError("output_tokens must be positive")
            if self.samples_per_length <= 0:
                raise ValueError("samples_per_length must be positive")

        elif self.type == "decode":
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
    def create_from_cli_args(cls) -> list["MicrobenchmarkConfig"]:
        """Create MicrobenchmarkConfig instances from CLI args or --from-file."""
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        instances = []
        for flat_config in flat_configs:
            instance = flat_config.reconstruct_original_dataclass()
            instances.append(instance)
        return instances
