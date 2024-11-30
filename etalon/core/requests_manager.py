from typing import Any, List
from threading import Thread, Lock
from queue import Queue

from etalon.core.llm_clients import construct_client
from etalon.core.request_config import RequestConfig


class AsyncRequestsManager:
    """Manages requests for single LLM API client."""

    def __init__(
        self,
        client_id: int,
        model: str,
        tokenizer_name: str,
        llm_api: str,
        max_concurrent_requests: int,
    ):
        self.max_concurrent_requests = max_concurrent_requests
        self.requests_queue = Queue()
        self.result_lock = Lock()
        self.results = []
        # just create a single client per manager
        self.llm_client = construct_client(
            model_name=model,
            tokenizer_name=tokenizer_name,
            llm_api=llm_api,
        )
        self.client_id = client_id

    def start_tasks(self):
        """Starts the tasks to handle requests.

        Returns:
            None
        """
        self.client_threads = [
            Thread(target=self.process_requests)
            for i in range(self.max_concurrent_requests)
        ]

        for thread in self.client_threads:
            thread.start()

    def process_requests(self) -> None:
        while True:
            request_config = self.requests_queue.get()
            if request_config is None:
                break
            result = self.llm_client.send_llm_request(request_config)
            with self.result_lock:
                self.results.append(result)

    def launch_requests(self, request_config: RequestConfig) -> List[Any]:
        """Launch requests to the LLM API.

        Args:
            request_config: The configuration for the request.

        """
        self.requests_queue.put(request_config)

    def get_results(self) -> List[Any]:
        """Return results that are ready from completed requests.

        Returns:
            A list of results that are ready.

        """
        with self.result_lock:
            curr_results = self.results
            self.results = []
        return curr_results

    def complete_tasks(self):
        """Waits for all tasks to complete.

        Returns:
            None
        """
        for _ in range(self.max_concurrent_requests):
            self.requests_queue.put(None)

        for thread in self.client_threads:
            thread.join()

