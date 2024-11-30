import time
import socket
from typing import Any, List

import ray
from ray.util import ActorPool

from etalon.core.request_config import RequestConfig
from etalon.core.requests_manager import AsyncRequestsManager

def get_ip() -> str:
    return socket.gethostbyname(socket.gethostname())


class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(
        self,
        model: str,
        tokenizer_name: str,
        llm_api: str,
        num_clients: int,
        num_concurrent_requests_per_client: int,
    ):
        self.actors = []
        worker_class = ray.remote(num_cpus=1)(AsyncRequestsManager).options(
            resources={f"node:{get_ip()}": 0.01}
        )

        for client_id in range(num_clients):
            self.actors.append(
                worker_class.remote(
                    client_id=client_id,
                    model=model,
                    tokenizer_name=tokenizer_name,
                    llm_api=llm_api,
                    max_concurrent_requests=num_concurrent_requests_per_client,
                )
            )
        self.llm_client_pool = ActorPool(self.actors)

    def start(self) -> None:
        """Starts the tasks on each actor to handle requests.

        Returns:
            None

        """
        for actor in self.actors:
            actor.start_tasks.remote()

    def launch_requests(self, request_config: RequestConfig) -> None:
        """Launch requests to the LLM API.

        Args:
            request_config: The configuration for the request.

        """
        self.llm_client_pool.submit(
            lambda actor, _request_config: actor.launch_requests.remote(
                _request_config
            ),
            request_config,
        )

    def is_free(self) -> bool:
        """Check if the pool of actors is free.

        Returns:
            True if the pool of actors is free, False otherwise.

        """
        return self.llm_client_pool.has_free()

    def free_pool(self, block: bool = False) -> None:
        """Frees the pool of actors for the next batch of requests.

        Args:
            block: Whether to block until a result is ready.

        Returns:
            None

        """
        if not block:
            while self.llm_client_pool.has_next():
                self.llm_client_pool.get_next_unordered()
        else:
            while len(self.llm_client_pool._pending_submits) > 0:
                time.sleep(0.1)
                pass
            while self.llm_client_pool.has_next():
                self.llm_client_pool.get_next_unordered()

    def complete_tasks(self) -> None:
        """Complete all tasks"""
        self.free_pool(block=True)
        ray.get([actor.complete_tasks.remote() for actor in self.actors])

    def collect_results(self) -> List[Any]:
        """Collect results from the actors.

        Returns:
            A list of results from the actors.

        """
        output = ray.get([
            actor.get_results.remote() for actor in self.actors
        ])
        output = [item for sublist in output for item in sublist]
        return output
