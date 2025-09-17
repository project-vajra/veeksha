import json
import os
from dataclasses import field
from typing import Any, Dict

from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.utils import dataclass_to_dict, get_config_hash
from veeksha.constants.configuration_constants import DEFAULT_SEED
from veeksha.logger import init_logger

logger = init_logger(__name__)

CONFIG_FILE_NAME = "config.json"


@frozen_dataclass
class BaseEntrypointConfig:
    """Configuration for microbenchmark profilers. This runs prefill and/or decode profiling
    based on the enabled flags in each section."""

    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Seed for the random number generator for capacity search."},
    )

    output_dir: str = field(
        default="entrypoint_experiments",
        metadata={"help": "Output directory for entrypoint results."},
    )

    @classmethod
    def create_from_cli_args(cls):
        """Create config instances from CLI

        Returns:
            List of config instances (single or
            multiple configs if YAML expands to multiple configurations)
        """
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        instances = []
        for flat_config in flat_configs:
            instance = flat_config.reconstruct_original_dataclass()
            object.__setattr__(instance, "__flat_config__", flat_config)
            instances.append(instance)
        return instances


    def get_hash(self) -> str:
        return get_config_hash(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        flat_config = getattr(self, "__flat_config__", None)
        if flat_config is None:
            logger.debug("Flat config not found or is None. Using dataclass_to_dict.")
            return dataclass_to_dict(self)

        return flat_config.__dict__  # type: ignore

    def write_config_to_file(self) -> None:
        config_dict = self.to_dict()
        output_path = os.path.join(self.output_dir, CONFIG_FILE_NAME)
        os.makedirs(self.output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=4)
