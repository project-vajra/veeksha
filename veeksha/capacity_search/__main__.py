import multiprocessing
import platform
import time
from typing import List

from veeksha.capacity_search.search_manager import SearchManager
from veeksha.config.capacity_search_config import CapacitySearchConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


def run():
    logger.info("Starting capacity search")
    capacity_search_configs: List[CapacitySearchConfig] = (
        CapacitySearchConfig.create_from_cli_args()
    )

    search_manager = SearchManager(capacity_search_configs)
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
