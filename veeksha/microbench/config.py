"""Microbenchmark configuration with inheritance."""

import sys
from argparse import ArgumentParser
from dataclasses import field
from typing import Any

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.utils import load_yaml_config


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

    @staticmethod
    def create_from_cli_args() -> list["BaseMicrobenchmarkConfig"]:
        """Parse CLI args, determine type from YAML, and construct typed configs."""
        pre_parser = ArgumentParser(add_help=False)
        pre_parser.add_argument("--microbenchmark-config-from-file", default=None)
        pre_args, _ = pre_parser.parse_known_args()

        if pre_args.microbenchmark_config_from_file is None:
            print("error: --microbenchmark-config-from-file is required", file=sys.stderr)
            sys.exit(1)

        yaml_config = load_yaml_config(pre_args.microbenchmark_config_from_file)

        configs: list[dict[str, Any]]
        if isinstance(yaml_config, list):
            configs = yaml_config
        else:
            configs = [yaml_config]

        instances: list[BaseMicrobenchmarkConfig] = []
        for raw in configs:
            assert isinstance(raw, dict), f"expected dict in YAML config, got {type(raw)}"
            type_name = raw.pop("type", None)
            if type_name is None:
                print("error: 'type' field is required in microbenchmark config", file=sys.stderr)
                sys.exit(1)
            config_cls = _TYPE_TO_CONFIG.get(type_name)
            if config_cls is None:
                valid = ", ".join(sorted(_TYPE_TO_CONFIG))
                print(
                    f"error: unknown microbenchmark type '{type_name}'. Valid types: {valid}",
                    file=sys.stderr,
                )
                sys.exit(1)
            instances.append(config_cls(**raw))

        return instances


@frozen_dataclass
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


@frozen_dataclass
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


_TYPE_TO_CONFIG: dict[str, type[BaseMicrobenchmarkConfig]] = {
    "prefill": PrefillMicrobenchmarkConfig,
    "decode": DecodeMicrobenchmarkConfig,
}
