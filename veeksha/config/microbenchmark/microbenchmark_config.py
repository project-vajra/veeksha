import json
import os
from dataclasses import field
from typing import Optional

from veeksha.config.benchmark_config import BenchmarkConfig
from veeksha.config.client_config import ClientConfig
from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.metrics_config import MetricsConfig
from veeksha.config.microbenchmark import (
    BaseMicrobenchmarkProbeConfig,
    PrefillProbeConfig,
)
from veeksha.constants.configuration_constants import DEFAULT_SEED
from veeksha.logger import init_logger

logger = init_logger(__name__)


@frozen_dataclass
class MicrobenchmarkConfig:
    """Configuration for microbenchmark profilers. This runs prefill and/or decode profiling
    based on the enabled flags in each section."""

    

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
