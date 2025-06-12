from functools import partial
from multiprocessing import Pool
import multiprocessing

from veeksha.capacity_search.capacity_search import CapacitySearch
from veeksha.config.config import BenchmarkConfig, CapacitySearchConfig
from veeksha.logger import init_logger
from typing import List, Dict

logger = init_logger(__name__)


def run_search(
    benchmark_config: BenchmarkConfig,
    capacity_search_config: Dict,
):
    capacity_search = CapacitySearch(
        CapacitySearchConfig(**capacity_search_config),
        benchmark_config,
    )
    return capacity_search.search()


def init_worker():
    # Make the current process non-daemon
    current = multiprocessing.current_process()
    current._config['daemon'] = False

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
        logger.info(f"Running {num_jobs} jobs sequentially")

        run_search_partial = partial(
            run_search,
            capacity_search_config=self.capacity_search_config.to_dict(),
        )
        all_results = [run_search_partial(cfg) for cfg in self.benchmark_configs]

        return all_results
