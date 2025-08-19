import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import is_dataclass, fields
import os
import random
from typing import List, Any

import wandb

from veeksha.capacity_search.capacity_search import CapacitySearch
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)

<<<<<<< HEAD
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
=======
def run_search(config: CapacitySearchConfig):
    """Run a single capacity search with the given config."""
    capacity_search = CapacitySearch(config)
>>>>>>> 6ffb5d5 (Fix pickling problem)
    return capacity_search.search()

class SearchManager:
    def __init__(self, capacity_search_configs: List[CapacitySearchConfig]):
        self.capacity_search_configs = [
            self._deep_sanitize(cfg) for cfg in capacity_search_configs
        ]

    def _run_sequential_for_endpoint(self, configs_for_endpoint: List[CapacitySearchConfig]):
        """Runs the search for a list of configs sequentially for a single endpoint."""
        endpoint = configs_for_endpoint[0].benchmark_config.api_url
        logger.info(f"Running {len(configs_for_endpoint)} jobs sequentially for endpoint '{endpoint}'")
        return [run_search(cfg) for cfg in configs_for_endpoint]

    def run(self):
        """Run all capacity searches with parallel execution per endpoint."""
        grouped_configs = collections.defaultdict(list)
        for cfg in self.capacity_search_configs:
            grouped_configs[cfg.benchmark_config.api_url].append(cfg)

        num_parallel_jobs = len(grouped_configs)
        logger.info(f"Running {num_parallel_jobs} job groups in parallel.")

        with ProcessPoolExecutor(max_workers=num_parallel_jobs) as executor:
            future_to_endpoint = {
                executor.submit(self._run_sequential_for_endpoint, configs): endpoint
                for endpoint, configs in grouped_configs.items()
            }

            all_results = []
            for future in as_completed(future_to_endpoint):
                endpoint = future_to_endpoint[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Endpoint '{endpoint}' generated an exception: {e}")

        return all_results

    def _deep_sanitize(self, obj: Any) -> Any:
        """Recursively remove unpicklable attributes from dataclass instances."""
        if is_dataclass(obj):
            # Create new instance with only the public fields
            return type(obj)(**{
                field.name: self._deep_sanitize(getattr(obj, field.name))
                for field in fields(obj)
                if not field.name.startswith('__')
            })
        elif isinstance(obj, list):
            return [self._deep_sanitize(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: self._deep_sanitize(v) for k, v in obj.items()}
        return obj