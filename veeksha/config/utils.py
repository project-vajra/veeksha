import hashlib
import importlib.resources
import json
import logging
import os
import time
from importlib.resources.abc import Traversable
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def dict_to_args(class_dict):
    args = []
    for key, value in class_dict.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
                else:
                    args.append(f"--no-{key}")
            else:
                args.append(f"--{key} {value}")
    return " ".join(args)


def get_trace_file_path(filename: str) -> Traversable:
    """
    Resolves the path to a data file within the package's processed_traces directory.

    Args:
        filename: The name of the file in veeksha.data.processed_traces.

    Returns:
        A PosixPath object representing the path to the data file.
    """
    return importlib.resources.files("veeksha.data.processed_traces").joinpath(filename)


def get_config_hash(config_dict: Dict[str, Any]) -> str:
    """Return a stable 8-char hash for config dictionaries.

    - Recursively removes volatile keys that can vary between runs
      (e.g., output directories or wandb runtime values).
    - Uses JSON with sorted keys to ensure deterministic ordering.
    """

    VOLATILE_KEYS = {
        "output_dir",
        "wandb_run_name",
        "wandb_sweep_id",
        "wandb_group",
        "__flat_config__",
    }

    def scrub(obj):
        if isinstance(obj, dict):
            return {k: scrub(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
        if isinstance(obj, list):
            return [scrub(i) for i in obj]
        return obj

    scrubbed = scrub(config_dict)
    stable_json = json.dumps(scrubbed, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2s(stable_json.encode()).hexdigest()[:8]


def _build_unique_output_dir(root: str, model_name: str, config_hash: str) -> str:
    """Return a unique timestamped output directory path.

    Format: <root>/<model>-<hash>-<timestamp>
    """
    timestamp = (
        time.strftime("%Y%m%d-%H%M%S", time.localtime())
        + f"-{int(time.time()*1000)%1000:03d}"
    )
    return os.path.join(root, f"{model_name}-{config_hash}-{timestamp}")


def prepare_benchmark_output_dir(benchmark_config) -> None:
    """Create a unique output subdirectory and persist config.
    - Always create a unique subdirectory under `metrics_config.output_dir`,
      named with model and config-hash plus a high-entropy timestamp.
    - Save both `config.json` and `config.yml` in the final output directory.
    """
    from vidhi import dataclass_to_dict

    current_output_dir = benchmark_config.metrics_config.output_dir
    existing_config_path = os.path.join(current_output_dir, "config.json")
    if os.path.isfile(existing_config_path):
        logger.debug(
            "Benchmark output directory already prepared at %s; skipping regeneration",
            current_output_dir,
        )
        return

    base_output_dir = benchmark_config.metrics_config.output_dir
    model_name = benchmark_config.client_config.model.split("/")[-1]

    config_as_dict = dataclass_to_dict(benchmark_config)
    assert isinstance(
        config_as_dict, dict
    ), f"Expected dict, got {type(config_as_dict)}"
    cfg_hash = get_config_hash(config_as_dict)
    unique_dir = _build_unique_output_dir(base_output_dir, model_name, cfg_hash)
    object.__setattr__(benchmark_config.metrics_config, "output_dir", unique_dir)
    os.makedirs(benchmark_config.metrics_config.output_dir, exist_ok=True)

    # write config.json
    with open(
        os.path.join(benchmark_config.metrics_config.output_dir, "config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config_as_dict, f, indent=4)

    # also write the yml file for rapid reproducibility
    with open(
        os.path.join(benchmark_config.metrics_config.output_dir, "config.yml"),
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            config_as_dict,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
