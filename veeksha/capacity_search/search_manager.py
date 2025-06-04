from functools import partial
from multiprocessing import Pool

from veeksha.capacity_search.capacity_search import CapacitySearch
from veeksha.config.config import CapacitySearchConfig, BenchmarkConfig
from veeksha.logger import init_logger
from typing import List

logger = init_logger(__name__)


def run_search(
    capacity_search_config: CapacitySearchConfig,
    benchmark_config: BenchmarkConfig,
):
    capacity_search = CapacitySearch(
        capacity_search_config,
        benchmark_config,
    )
    return capacity_search.search()


class SearchManager:
    def __init__(
        self,
        capacity_search_config: CapacitySearchConfig,
        benchmark_configs: List[BenchmarkConfig],
    ):
        self.capacity_search_config = capacity_search_config
        self.benchmark_configs = benchmark_configs

    def run(self):
        num_jobs = len(self.benchmark_configs)
        logger.info(f"Running {num_jobs} jobs")

        with Pool(processes=num_jobs) as capacity_search_pool:
            run_search_partial = partial(run_search, self.capacity_search_config)
            all_results = capacity_search_pool.map(run_search_partial, self.benchmark_configs)

        return all_results
