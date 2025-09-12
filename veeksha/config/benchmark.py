import json
import os
from dataclasses import field
from datetime import datetime
from typing import Optional

from veeksha.config.client import ClientConfig
from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
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
from veeksha.config.utils import dataclass_to_dict, get_config_hash
from veeksha.constants.configuration_constants import DEFAULT_SEED
from veeksha.logger import init_logger
from veeksha.types import RequestGeneratorType

logger = init_logger(__name__)


@frozen_dataclass(allow_from_file=True)
class BenchmarkConfig:
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
    client_config: ClientConfig = field(
        default_factory=ClientConfig,
        metadata={"help": "The client configuration for the benchmark."},
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "The metrics configuration for the benchmark."},
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig,
        metadata={"help": "The request generator configuration for the benchmark."},
    )

    def __post_init__(self):
        config_hash = get_config_hash(dataclass_to_dict(self))
        model_name = self.client_config.model.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_output_dir = os.path.join(
            self.metrics_config.output_dir, f"{model_name}-{config_hash}-{timestamp}"
        )

        # frozen dataclass
        object.__setattr__(self.metrics_config, "output_dir", unique_output_dir)

        if not os.path.exists(self.metrics_config.output_dir):
            os.makedirs(self.metrics_config.output_dir, exist_ok=True)

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
        """Create BenchmarkConfig instances from CLI

        Returns:
            List of BenchmarkConfig instances (single or
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
