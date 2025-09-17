import os
import random
from typing import List

import wandb

from veeksha.capacity_search.capacity_search import CapacitySearch
from veeksha.config.capacity_search_config import CapacitySearchConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


def run_search(
    config: CapacitySearchConfig,
):
    random.seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)

    capacity_search = CapacitySearch(config)
    return capacity_search.search()


# TODO implement parallel jobs if they have different servers
class SearchManager:
    def __init__(
        self,
        configs: List[CapacitySearchConfig],
    ):
        self.configs = configs

    def run(self):
        num_jobs = len(self.configs)
        logger.info(f"Running {num_jobs} jobs sequentially")
        logger.info(f"Capacity search configs:")
        for i, cfg in enumerate(self.configs):
            logger.info(f"- {i}: {cfg} \n")

        all_results = [
            run_search(cfg_params) for cfg_params in self.configs
        ]

        return all_results
