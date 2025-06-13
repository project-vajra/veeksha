import multiprocessing
from functools import partial
from typing import Dict, List

from veeksha.capacity_search.capacity_search import CapacitySearch
from veeksha.config.config import CapacitySearchConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


def run_search(
    benchmark_config_params: Dict,
    capacity_search_config: CapacitySearchConfig,
):
    capacity_search = CapacitySearch(
        capacity_search_config,
        benchmark_config_params,
    )
    return capacity_search.search()


def init_worker():
    # Make the current process non-daemon
    current = multiprocessing.current_process()
    current._config["daemon"] = False


class SearchManager:
    def __init__(
        self,
        capacity_search_config: CapacitySearchConfig,
        benchmark_configs_params: List[Dict],
    ):
        self.capacity_search_config = capacity_search_config
        self.benchmark_configs_params = benchmark_configs_params

    def run(self):
        num_jobs = len(self.benchmark_configs_params)
        logger.info(f"Running {num_jobs} jobs sequentially")

        run_search_partial = partial(
            run_search,
            capacity_search_config=self.capacity_search_config,
        )
        all_results = [
            run_search_partial(cfg_params)
            for cfg_params in self.benchmark_configs_params
        ]

        return all_results
