import json
import os
from dataclasses import field
from typing import Optional

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import ClientConfig
from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.metrics import MetricsConfig
from veeksha.config.microbenchmark.base_microbenchmark import BaseMicrobenchmarkProbeConfig
from veeksha.config.microbenchmark.prefill_probe import PrefillProbeConfig
from veeksha.constants.configuration_constants import DEFAULT_SEED
from veeksha.logger import init_logger

logger = init_logger(__name__)


@frozen_dataclass(allow_from_file=True)
class MicrobenchmarkConfig:
    """Configuration for microbenchmark profilers. This runs prefill and/or decode profiling
    based on the enabled flags in each section."""

    model: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "The model to use for this microbenchmark."},
    )
    api_url: str = field(
        default="http://localhost:30000/v1",
        metadata={"help": "The API URL for the benchmark."},
    )
    api_key: str = field(
        default="token-abc123",
        metadata={"help": "The API key for the benchmark."},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for this microbenchmark. By default, the tokenizer is inferred from the model."
        },
    )
    additional_sampling_params: str = field(
        default="{}",
        metadata={"help": "Additional sampling params."},
    )
    timeout: int = field(
        default=1200,
        metadata={"help": "The amount of time to run each profiling run for."},
    )
    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Seed for the random number generator."},
    )
    output_dir: str = field(
        default="microbenchmark_experiments",
        metadata={"help": "Output directory for microbenchmark results."},
    )
    probe_config: BaseMicrobenchmarkProbeConfig = field(
        default_factory=PrefillProbeConfig,
        metadata={
            "help": "Polymorphic microbenchmark probe configuration (e.g., prefill, decode).",
        },
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb project."},
    )

    def create_benchmark_config(self, output_dir: str = None) -> BenchmarkConfig:
        """Create a BenchmarkConfig from microbenchmark settings.

        Args:
            output_dir: Override output directory. If None, uses self.output_dir

        Returns:
            BenchmarkConfig built from microbenchmark settings
        """
        return BenchmarkConfig(
            seed=self.seed,
            timeout=self.timeout,
            api_url=self.api_url,
            api_key=self.api_key,
            client_config=ClientConfig(
                model=self.model,
                tokenizer=self.tokenizer,
                additional_sampling_params=self.additional_sampling_params,
            ),
            metrics_config=MetricsConfig(
                output_dir=output_dir or self.output_dir,
            ),
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
