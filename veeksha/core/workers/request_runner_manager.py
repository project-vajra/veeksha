"""Manager for async request runner worker pool."""

from queue import Queue
from threading import Event, Thread
from typing import List, Optional

from veeksha.config.client import ClientConfig
from veeksha.core.context import WorkerContext
from veeksha.core.workers.request_runner_worker import RequestRunnerWorker, InputOutputWriter
from veeksha.logger import init_logger
from collections import defaultdict

logger = init_logger(__name__)


class RequestRunnerManager:
    """Manages a pool of async request runner workers."""

    def __init__(
        self,
        client_config: ClientConfig,
        input_queue: Queue,
        output_queue: Queue,
        num_threads: int,
        input_output_writer: Optional[InputOutputWriter] = None,
    ):
        self.client_config = client_config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.num_threads = num_threads
        self.input_output_writer = input_output_writer
        self.workers: List[Thread] = []
        self.worker_contexts: List[WorkerContext] = []
        self.stop_event = Event()

        self._chat_history = defaultdict(list)

    def start(self) -> None:
        """Start async worker threads with uvloop event loops."""
        logger.info(f"Starting {self.num_threads} async worker threads with uvloop")

        for i in range(self.num_threads):
            # Create worker context with load tracking
            worker_context = WorkerContext(worker_id=i, stop_event=self.stop_event)
            self.worker_contexts.append(worker_context)

            # Create worker instance
            worker_instance = RequestRunnerWorker(
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                worker_context=worker_context,
                client_config=self.client_config,
                input_output_writer=self.input_output_writer,
                chat_history=self._chat_history,
            )

            # Create thread running async worker with uvloop
            worker = Thread(
                target=worker_instance.run,
                name=f"async-worker-{i}",
                daemon=False,
            )
            self.workers.append(worker)
            worker.start()

    def get_worker_count(self) -> int:
        """Return the number of worker threads."""
        return len(self.workers)

    def get_worker_contexts(self) -> List[WorkerContext]:
        """Return worker contexts for power-of-two load balancing."""
        return self.worker_contexts

    def complete_tasks(self) -> None:
        """Signal worker threads to complete their tasks and exit."""
        # Send one sentinel value per worker
        for _ in range(len(self.workers)):
            self.input_queue.put(None)

        # Set stop event (shared by all workers)
        self.stop_event.set()

    def wait_for_workers(self) -> None:
        """Wait for all worker threads to complete their tasks and exit."""
        self.complete_tasks()
        for worker in self.workers:
            worker.join()
