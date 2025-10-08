import multiprocessing
import os
import platform
import time
from typing import List

from veeksha.capacity_search.search_manager import SearchManager
from veeksha.config.capacity_search import CapacitySearchConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


def run():
    logger.info("Starting capacity search")
    capacity_search_configs: List[CapacitySearchConfig] = (
        CapacitySearchConfig.create_from_cli_args()
    )

    # Check if dashboard is enabled
    has_dashboard_enabled = any(
        config.get_dashboard_config().enabled
        for config in capacity_search_configs
    )
    if has_dashboard_enabled:
        # Set environment variable to suppress console logging in child processes
        os.environ["VEEKSHA_SUPPRESS_CONSOLE_LOGS"] = "1"
        # Suppress tokenizers parallelism warning when forking processes
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Remove stream handlers from loggers (but don't redirect stdout/stderr)
        # This allows the TUI to start properly and LogCapture to buffer logs
        import logging as log_module
        
        # Remove handlers from root logger
        root_logger = log_module.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, log_module.StreamHandler):
                root_logger.removeHandler(handler)
        
        # Remove handlers from veeksha logger specifically (which has its own handler)
        veeksha_logger = log_module.getLogger("veeksha")
        for handler in veeksha_logger.handlers[:]:
            if isinstance(handler, log_module.StreamHandler):
                veeksha_logger.removeHandler(handler)

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
