import os
from typing import List

import wandb

from veeksha.capacity_search.capacity_search import CapacitySearch
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


def run_search(
    capacity_search_config: CapacitySearchConfig,
):
    if (
        capacity_search_config.wandb_project
        and capacity_search_config.enable_wandb_sweep
    ):
        assert (
            capacity_search_config.wandb_sweep_id
            or capacity_search_config.wandb_sweep_name
        ), "wandb-sweep-name/id is required with wandb-project"

    os.makedirs(capacity_search_config.output_dir, exist_ok=True)

    if (
        capacity_search_config.wandb_project
        and capacity_search_config.enable_wandb_sweep
        and not capacity_search_config.wandb_sweep_id
    ):
        capacity_search_config.wandb_sweep_id = wandb.sweep(
            capacity_search_config.to_dict(),
            project=capacity_search_config.wandb_project,
        )
        # required so that wandb doesn't delay flush of child logs
        wandb.finish(quiet=True)

    capacity_search = CapacitySearch(capacity_search_config)
    return capacity_search.search()


# TODO implement parallel jobs if they have different servers
class SearchManager:
    def __init__(
        self,
        capacity_search_configs: List[CapacitySearchConfig],
    ):
        self.capacity_search_configs = capacity_search_configs

    def run(self):
        num_jobs = len(self.capacity_search_configs)
        logger.info(f"Running {num_jobs} jobs sequentially")
        logger.info(f"Capacity search configs:")
        for i, cfg in enumerate(self.capacity_search_configs):
            logger.info(f"- {i}: {cfg} \n")

        all_results = [
            run_search(cfg_params) for cfg_params in self.capacity_search_configs
        ]

        return all_results
