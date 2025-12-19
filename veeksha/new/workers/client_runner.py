"""Client runner manager for async request execution."""

import asyncio
import threading
from queue import Empty, Queue
from typing import List

from veeksha.logger import init_logger
from veeksha.new.client.base import BaseLLMClient

logger = init_logger(__name__)


QUEUE_GET_TIMEOUT_S = 0.1


class ClientWorker:
    """Async client worker that processes requests from an input queue."""

    def __init__(
        self,
        worker_id: int,
        client: BaseLLMClient,
        input_queue: Queue,
        output_queue: Queue,
        stop_event: threading.Event,
    ):
        self.worker_id = worker_id
        self.client = client
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self) -> None:
        """Run the async event loop for this worker."""
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        """Async main loop."""
        logger.debug("Client worker %d starting", self.worker_id)

        while not self.stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=QUEUE_GET_TIMEOUT_S)
            except Empty:
                continue

            if item is None:  # Sentinel
                break

            request, session_id, session_size, dispatched_at = item

            result = await self.client.send_request(
                request=request,
                session_id=session_id,
                session_total_requests=session_size,
            )

            # Update dispatched_at from caller (more accurate)
            result.dispatched_at = dispatched_at

            # Put result in output queue
            self.output_queue.put(result)

        logger.debug("Client worker %d exiting", self.worker_id)


class ClientRunnerManager:
    """Manager for a pool of async client worker threads."""

    def __init__(
        self,
        client: BaseLLMClient,
        input_queues: List[Queue],
        output_queue: Queue,
        stop_event: threading.Event,
    ):
        """Initialize the client runner manager.

        Args:
            client: LLM client to use for requests
            input_queues: One input queue per worker
            output_queue: Shared output queue for results
            stop_event: Stop event for graceful shutdown
        """
        self.client = client
        self.input_queues = input_queues
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.workers: List[ClientWorker] = []
        self.threads: List[threading.Thread] = []

    def start(self) -> None:
        """Start all client worker threads."""
        for i, queue in enumerate(self.input_queues):
            worker = ClientWorker(
                worker_id=i,
                client=self.client,
                input_queue=queue,
                output_queue=self.output_queue,
                stop_event=self.stop_event,
            )
            self.workers.append(worker)

            thread = threading.Thread(
                target=worker.run,
                name=f"client-worker-{i}",
                daemon=True,
            )
            self.threads.append(thread)
            thread.start()

        logger.info("Started %d client worker threads", len(self.threads))

    def stop(self) -> None:
        """Signal workers to stop."""
        self.stop_event.set()
        for queue in self.input_queues:
            queue.put(None)  # Sentinel

    def wait(self) -> None:
        """Wait for all worker threads to finish."""
        for thread in self.threads:
            thread.join()
        logger.debug("All client worker threads joined")

    def get_worker_count(self) -> int:
        """Return number of workers."""
        return len(self.workers)
