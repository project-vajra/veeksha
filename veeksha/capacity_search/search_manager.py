import os
from typing import List, Optional

import wandb

from veeksha.capacity_search.capacity_search import CapacitySearch, SearchResult
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
        wandb.finish()

    # Initialize dashboard if enabled
    dashboard_cfg = capacity_search_config.get_dashboard_config()
    if dashboard_cfg.enabled:
        from veeksha.dashboard.handler import init_dashboard_event_processor

        dashboard_state = init_dashboard_event_processor(
            enabled=True,
            enable_frontend=False,
            max_queue_size=dashboard_cfg.max_queue_size,
            max_live_requests=dashboard_cfg.max_live_requests,
        )

        # Run capacity search in background thread, TUI in main thread
        import threading

        result_container: dict[str, Optional[SearchResult]] = {"result": None}

        def run_search_thread():
            capacity_search = CapacitySearch(capacity_search_config)
            result_container["result"] = capacity_search.search()

        search_thread = threading.Thread(target=run_search_thread, daemon=False)
        search_thread.start()

        # Run TUI in main thread
        from veeksha.dashboard.tui_dashboard import run_dashboard_tui

        if dashboard_state:
            run_dashboard_tui(dashboard_state)

        # Wait for search to complete
        search_thread.join()

        return result_container["result"]
    else:
        # No dashboard - run normally
    # Check if server_config is specified for automatic server management
    if capacity_search_config.benchmark_config.server_config is not None:
        from veeksha.orchestration import managed_server

        logger.info("Server config detected, launching managed server for capacity search")
        with managed_server(capacity_search_config.benchmark_config.server_config) as server_info:
            logger.info(f"Server ready at {server_info['api_base']}")
            capacity_search = CapacitySearch(capacity_search_config)
            return capacity_search.search()
    else:
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
