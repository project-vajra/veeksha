import json
import os
from dataclasses import field
from typing import Optional

import yaml  # type: ignore

from veeksha.config.client import ClientConfig
from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.deadline import DeadlineConfig
from veeksha.config.generators.request_generator.base_generator import (
    BaseRequestGeneratorConfig,
)
from veeksha.config.generators.request_generator.lmeval_generator import (
    LmevalRequestGeneratorConfig,
)
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.config.metrics import MetricsConfig
from veeksha.config.prefill_profiler import PrefillProfilerConfig
from veeksha.config.utils import create_class_from_dict, dataclass_to_dict, expand_dict
from veeksha.constants.configuration_constants import DEFAULT_SEED
from veeksha.logger import init_logger
from veeksha.types import RequestGeneratorType

logger = init_logger(__name__)


@frozen_dataclass
class BenchmarkConfig:
    # TODO seed is set once in the benchmarkconfig and propagated to all nested configs
    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Seed for the random number generator."},
    )
    timeout: int = field(
        default=1200,
        metadata={"help": "The amount of time to run the load test for."},
    )
    max_completed_requests: int = field(
        default=10,
        metadata={
            "help": "The number of requests to complete before finishing the test. Note "
            "that its possible for the test to timeout first."
        },
    )
    api_url: Optional[str] = field(
        default="http://localhost:8000/v1",
        metadata={"help": "The API URL for the benchmark."},
    )
    api_key: Optional[str] = field(
        default="token-abc123",
        metadata={"help": "The API key for the benchmark."},
    )
    benchmark_config_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to YAML configuration file for the benchmark. If it's provided, no other parameters will be used."
        },
    )
    client_config: ClientConfig = field(
        default_factory=ClientConfig,
        metadata={"help": "The client configuration for the benchmark."},
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "The metrics configuration for the benchmark."},
    )
    deadline_config: DeadlineConfig = field(
        default_factory=DeadlineConfig,
        metadata={"help": "The deadline configuration for the benchmark."},
    )
    prefill_profiler_config: PrefillProfilerConfig = field(
        default_factory=PrefillProfilerConfig,
        metadata={"help": "The prefill profiler configuration for the benchmark."},
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig,
        metadata={"help": "The request generator configuration for the benchmark."},
    )

    # TODO move this away
    def __post_init__(self):
        if not os.path.exists(self.metrics_config.output_dir):
            os.makedirs(self.metrics_config.output_dir)

        if self.prefill_profiler_config.use_predictions_for_ttft:
            self.prefill_profiler_config.max_prefill_tokens_to_predict = max(
                self.prefill_profiler_config.max_prefill_tokens_to_predict,
                self.request_generator_config.max_tokens,
            )
            self.prefill_profiler_config.fill_predictions_array()

        if self.request_generator_config.get_type() == RequestGeneratorType.LMEVAL:
            logger.warning("Removing timeout for LMEval.")
            self.timeout = -1
            assert isinstance(
                self.request_generator_config, LmevalRequestGeneratorConfig
            )

            if self.request_generator_config.is_logit_based:
                self.client_config.llm_api = "openai_completions"
                self.client_config.address_append_value = "completions"
            else:
                self.client_config.llm_api = "openai_chat"
                self.client_config.address_append_value = "chat/completions"

        self.write_config_to_file()

    @classmethod
    def create_from_cli_args(cls):
        """Create BenchmarkConfig instances from CLI args or YAML file.

        Returns:
            List of BenchmarkConfig instances (single config for CLI args,
            multiple configs if YAML expands to multiple configurations)
        """
        import argparse

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--benchmark-config-file", type=str, default=None)
        known_args, _ = parser.parse_known_args()

        # If config_file is specified, load from YAML instead
        if known_args.benchmark_config_file:
            logger.info(
                f"Loading configuration from YAML file: {known_args.benchmark_config_file}"
            )
            return cls.create_from_yaml_file(known_args.benchmark_config_file)

        # Otherwise, use normal CLI args parsing and return as single-item list
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        instance.__flat_config__ = flat_config
        return [instance]

    @classmethod
    def create_from_yaml_file(cls, config_file_path: str):
        """Create BenchmarkConfig instances from a YAML configuration file.

        Returns:
            List of BenchmarkConfig instances (one for each expanded configuration)
        """
        with open(config_file_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        expanded_configs = expand_dict(yaml_config)

        logger.info(
            f"YAML config expanded to {len(expanded_configs)} configuration(s)."
        )

        instances = []
        for i, config_dict in enumerate(expanded_configs):
            instance = create_class_from_dict(cls, config_dict)
            # Use object.__setattr__ because this is a frozen dataclass
            object.__setattr__(instance, "__flat_config__", None)
            instances.append(instance)

        return instances

    @classmethod
    def create_flat_config(cls):
        instance = create_flat_dataclass(cls)
        instance.reconstruct_original_dataclass()
        instance.__flat_config__ = instance
        return

    def to_dict(self):
        if not hasattr(self, "__flat_config__") or self.__flat_config__ is None:
            logger.debug("Flat config not found or is None. Using dataclass_to_dict.")
            return dataclass_to_dict(self)

        return self.__flat_config__.__dict__  # type: ignore

    def write_config_to_file(self):
        config_dict = dataclass_to_dict(self)
        with open(
            os.path.join(f"{self.metrics_config.output_dir}", "config.json"), "w"
        ) as f:
            json.dump(config_dict, f, indent=4)
