import json
import os
from dataclasses import field
from typing import Optional, Any

import yaml  # type: ignore

from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.utils import create_class_from_dict
from veeksha.constants.configuration_constants import DEFAULT_SEED
from veeksha.logger import init_logger
from veeksha.config.slo import SLOSet, BaseSLO

logger = init_logger(__name__)


@frozen_dataclass
class CapacitySearchConfig:
    """Configuration for capacity search benchmark. This is a special benchmark that runs multiple benchmarks with different QPS and
    finds the maximum QPS that can be sustained given the deadline constraints."""

    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Seed for the random number generator for capacity search."},
    )
    start_qps: float = field(
        default=1,
        metadata={"help": "The starting QPS for the capacity search."},
    )
    num_qps_steps: int = field(
        default=10,
        metadata={"help": "The number of QPS steps for the capacity search."},
    )
    min_search_granularity: float = field(
        default=2.5,
        metadata={"help": "Minimum search granularity for capacity (%)"},
    )
    max_iterations: int = field(
        default=20,
        metadata={"help": "Maximum number of iterations for capacity search."},
    )
    output_dir: str = field(
        default="./veeksha/capacity_search/output",
        metadata={"help": "Output directory for capacity search."},
    )
    capsearch_config_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to YAML configuration file for the capacity search. If provided, no other parameters will be used."
        },
    )
    benchmark_config_file: str = field(
        default="./veeksha/capacity_search/config/default_config.yml",
        metadata={
            "help": "Path to benchmark config file. Benchmark config files can be expanded to multiple configurations."
        },
    )
    slos_config_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to SLOs configuration file (JSON or YAML). If provided, overrides legacy SLO settings."
        }
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb project for capacity search"},
    )
    enable_wandb_sweep: bool = field(
        default=False,
        metadata={"help": "Enable wandb sweep for capacity search"},
    )
    wandb_sweep_name: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb sweep name for capacity search"},
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb sweep id for capacity search"},
    )

    def get_slos(self) -> SLOSet:
        """Get or create SLOSet from this config."""

        if self.slos_config_file:
            # Load from external file
            if self.slos_config_file.endswith(
                ".json"
            ) or self.slos_config_file.endswith(".yaml"):
                with open(self.slos_config_file, "r") as f:
                    slo_dict = yaml.safe_load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {self.slos_config_file}"
                )

            # Handle polymorphic deserialization for slos
            slo_definitions = []
            for slo_def_dict in slo_dict.get("slos", []):
                slo_type_str = slo_def_dict.pop("type", None)
                if not slo_type_str:
                    raise ValueError(
                        "Each SLO definition in config file must have a 'type'"
                    )

                # Create the specific SLO definition class instance
                slo_class = BaseSLO.create_from_type(slo_type_str)
                slo_definitions.append(create_class_from_dict(slo_class, slo_def_dict))

            return SLOSet(
                slos=slo_definitions, require_all=slo_dict.get("require_all", True)
            )
        else:
            # Create from legacy config
            return SLOSet.from_capacity_search_config(self)

    @classmethod
    def create_from_cli_args(cls):
        """Create CapacitySearchConfig instance from CLI args or YAML file.
        Returns:
            CapacitySearchConfig instance
        """
        import argparse

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--capsearch-config-file", type=str, default=None)
        known_args, _ = parser.parse_known_args()

        # If config_file is specified, load from YAML instead
        if known_args.capsearch_config_file:
            logger.info(
                f"Loading configuration from YAML file: {known_args.capsearch_config_file}"
            )
            return cls.create_from_yaml_file(known_args.capsearch_config_file)

        # Otherwise, use normal CLI args parsing
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        object.__setattr__(instance, "__flat_config__", flat_config)
        return instance

    @classmethod
    def create_from_yaml_file(cls, config_file_path: str):
        """Create CapacitySearchConfig instance from a YAML configuration file.
        Returns:
            CapacitySearchConfig instance
        """
        with open(config_file_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        instance = create_class_from_dict(cls, yaml_config)
        # Use object.__setattr__ because this is a frozen dataclass
        object.__setattr__(instance, "__flat_config__", None)
        return instance

    def to_dict(self):
        return self.__dict__

    def write_config_to_file(self):
        config_dict = self.to_dict()
        output_path = os.path.join(self.output_dir, "config.json")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=4)
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write config to {output_path}: {e}")
