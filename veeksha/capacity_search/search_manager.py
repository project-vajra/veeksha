import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
from typing import List

import wandb

from veeksha.capacity_search.capacity_search import CapacitySearch
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)

def run_search(
    capacity_search_config: CapacitySearchConfig,
):
    random.seed(capacity_search_config.seed)
    if (
        capacity_search_config.wandb_project
        and capacity_search_config.enable_wandb_sweep
    ):
        assert (
            capacity_search_config.wandb_sweep_id
            or capacity_search_config.wandb_sweep_name
        ), "wandb-sweep-name/id is required with wandb-project"

    # assert (
    #     capacity_search_config.deadline_miss_rate_slo >= 0
    #     and capacity_search_config.deadline_miss_rate_slo <= 1
    # )

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


class SearchManager:
    def __init__(
        self,
        capacity_search_configs: List[CapacitySearchConfig],
    ):
        self.capacity_search_configs = capacity_search_configs

    def _run_sequential_for_endpoint(self, configs_for_endpoint):
        """Runs the search for a list of configs sequentially."""
        logger.info(
            f"Running {len(configs_for_endpoint)} jobs sequentially for endpoint "
            f"'{configs_for_endpoint[0].benchmark_config.api_url}'"
        )
        return [run_search(cfg) for cfg in configs_for_endpoint]

    def run(self):
<<<<<<< HEAD
        num_jobs = len(self.capacity_search_configs)
        logger.info(f"Running {num_jobs} jobs sequentially")
        logger.info(f"Capacity search configs:")
        for i, cfg in enumerate(self.capacity_search_configs):
            logger.info(f"- {i}: {cfg} \n")
=======
        grouped_configs = collections.defaultdict(list)
        for cfg in self.capacity_search_configs:
            grouped_configs[cfg.benchmark_config.api_url].append(cfg)
>>>>>>> ee6c210 (Add parallel search)

        print('grouped configs:')
        print(grouped_configs)

        all_results = []
        num_parallel_jobs = len(grouped_configs)
        logger.info(f"Running {num_parallel_jobs} job groups in parallel.")

        with ProcessPoolExecutor(max_workers=num_parallel_jobs) as executor:
            future_to_endpoint = {
                executor.submit(self._run_sequential_for_endpoint, configs): endpoint
                for endpoint, configs in grouped_configs.items()
            }

            print("future to endpiont:")
            print(future_to_endpoint)

            for future in as_completed(future_to_endpoint):
                endpoint = future_to_endpoint[future]
                try:
                    results_for_endpoint = future.result()
                    print(f"FINISHED ENDPOINT: {endpoint}")
                    all_results.extend(results_for_endpoint)
                except Exception as exc:
                    logger.error(f"Endpoint '{endpoint}' generated an exception: {exc}")
        
        return all_results
    