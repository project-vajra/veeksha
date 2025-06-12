import multiprocessing
import os
import platform
import random
import time
import json
import wandb
import yaml  # type: ignore

from veeksha.capacity_search.search_manager import SearchManager
from veeksha.config.config import CapacitySearchConfig, BenchmarkConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


def run():
    logger.info("Starting capacity search")
    capacity_search_config: CapacitySearchConfig = CapacitySearchConfig.create_from_cli_args()
    random.seed(capacity_search_config.seed)
    if capacity_search_config.wandb_project and capacity_search_config.enable_wandb_sweep:
        assert (
            capacity_search_config.wandb_sweep_id or
            capacity_search_config.wandb_sweep_name
        ), "wandb-sweep-name/id is required with wandb-project"

    benchmark_configs_yaml = yaml.safe_load(open(capacity_search_config.benchmark_config_file))
    benchmark_configs = BenchmarkConfig.generate_capacity_search_benchmark_configs(benchmark_configs_yaml)
    # TODO(chus): launch server support
    # if capacity_search_config.server_config_file:
    #     server_config = yaml.safe_load(open(capacity_search_config.server_config_file))
    # else:
    #     server_config = None
    #     logger.info("Server config not provided. Will not launch server.")

    assert capacity_search_config.deadline_miss_rate_slo >= 0 and capacity_search_config.deadline_miss_rate_slo <= 1

    os.makedirs(capacity_search_config.output_dir, exist_ok=True)

    # write configs to file
    # TODO(chus): when saving the config, we are saving the global config in a single cap search dir. 
    # We should save each cap search & benchmark config in its own dir, as they are different experiments.
    # TODO(chus): save config as yaml instead of json to be consistent with the input config format
    capacity_search_config_dict = capacity_search_config.to_dict()
    full_config = {"capacity_search": capacity_search_config_dict, **benchmark_configs_yaml}

    with open(os.path.join(capacity_search_config.output_dir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=4)

    if capacity_search_config.wandb_project and capacity_search_config.enable_wandb_sweep and not capacity_search_config.wandb_sweep_id:
        capacity_search_config.wandb_sweep_id = wandb.sweep(capacity_search_config.to_dict(), project=capacity_search_config.wandb_project)
        # required so that wandb doesn't delay flush of child logs
        wandb.finish(quiet=True)
    
    search_manager = SearchManager(capacity_search_config, benchmark_configs)
    start_time = time.time()
    all_results = search_manager.run()
    end_time = time.time()
    logger.info(f"Capacity search took time: {end_time - start_time}")


def main():
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    run()


if __name__ == "__main__":
    main()
