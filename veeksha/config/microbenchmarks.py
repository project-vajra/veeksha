import json
import os
from dataclasses import field
from typing import List, Optional

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.constants.configuration_constants import DEFAULT_SEED
from veeksha.logger import init_logger

logger = init_logger(__name__)


@frozen_dataclass(allow_from_file=True)
class PrefillProfilerSection:
    enabled: bool = field(
        default=True,
        metadata={"help": "Whether to run the prefill profiler."},
    )
    prefill_lengths: List[int] = field(
        default_factory=lambda: [2**i for i in range(8, 15)],
        metadata={"help": "The lengths to prefill the profiler with."},
    )


@frozen_dataclass(allow_from_file=True)
class DecodeProfilerSection:
    enabled: bool = field(
        default=False,
        metadata={"help": "Whether to run the decode profiler."},
    )
    context_lengths: List[int] = field(
        default_factory=lambda: [2**i for i in range(8, 15)],
        metadata={"help": "The lengths to decode the profiler with."},
    )
    engine_chunk_size: int = field(
        default=512,
        metadata={"help": "The chunk size the engine is running with."},
    )
    batch_sizes: List[int] = field(
        default_factory=lambda: [2**i for i in range(4, 8)],
        metadata={"help": "The batch sizes to decode the profiler with."},
    )
    engine_uses_mixed_batching: bool = field(
        default=False,
        metadata={"help": "Whether the engine uses mixed batching."},
    )


@frozen_dataclass(allow_from_file=True)
class MicrobenchmarkConfig:
    """Configuration for microbenchmark profilers. This runs prefill and/or decode profiling
    based on the enabled flags in each section."""

    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Seed for the random number generator for microbenchmarks."},
    )
    output_dir: str = field(
        default="./microbenchmark_output",
        metadata={"help": "Output directory for microbenchmark results."},
    )
    prefill_profiler: PrefillProfilerSection = field(
        default_factory=PrefillProfilerSection,
        metadata={"help": "Prefill profiler configuration."},
    )
    decode_profiler: DecodeProfilerSection = field(
        default_factory=DecodeProfilerSection,
        metadata={"help": "Decode profiler configuration."},
    )
    benchmark_config: BenchmarkConfig = field(
        default_factory=BenchmarkConfig,
        metadata={"help": "Base benchmark config for microbenchmarks."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb project for microbenchmarks"},
    )

    @classmethod
    def create_from_cli_args(cls):
        """Create MicrobenchmarkConfig instances from CLI

        Returns:
            List of MicrobenchmarkConfig instances (single or
            multiple configs if YAML expands to multiple configurations)
        """
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        instances = []
        for flat_config in flat_configs:
            instance = flat_config.reconstruct_original_dataclass()
            object.__setattr__(instance, "__flat_config__", flat_config)
            instances.append(instance)
        return instances

    def to_dict(self):
        from veeksha.config.utils import dataclass_to_dict
        return dataclass_to_dict(self)

    def write_config_to_file(self):
        config_dict = self.to_dict()
        output_path = os.path.join(self.output_dir, "config.json")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=4)
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write config to {output_path}: {e}")

