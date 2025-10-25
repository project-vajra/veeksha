from queue import Empty, Queue
from threading import Thread
from typing import Any, List, Optional

from veeksha.config.client import ClientConfig
from veeksha.core.llm_clients import construct_client
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


def _worker_thread(
    input_queue: Queue,
    output_queue: Queue,
    worker_id: int,
    client_config: ClientConfig,
) -> None:
    """Worker thread that processes requests from the queue.

    With GIL-free Python (python -Xgil=0), threads can achieve true parallelism
    similar to multiprocessing but with lower overhead.
    """
    logger.debug("Starting worker thread %s", worker_id)

    llm_client = construct_client(
        model_name=client_config.model,
        tokenizer_name=client_config.tokenizer or client_config.model,
        llm_api=client_config.llm_api,
    )

    while True:
        request_config = None
        try:
            try:
                request_config = input_queue.get(timeout=1.0)
            except Empty:
                continue  # Poll the queue again

            if request_config is None:
                # Sentinel value to signal shutdown
                break

            try:
                result = llm_client.send_llm_request(
                    request_config, client_config.request_timeout
                )
                output_queue.put(result)
            except Exception as e:
                logger.exception(
                    "send_llm_request failed for worker_id=%s",
                    worker_id,
                )
                _emit_error_result(
                    output_queue=output_queue,
                    e=e,
                    request_config=request_config,
                    worker_id=worker_id,
                )
                continue

        except Exception as e:
            logger.exception("Unexpected error in worker %s", worker_id)
            _emit_error_result(
                output_queue=output_queue,
                e=e,
                request_config=request_config,
                worker_id=worker_id,
            )
            continue

    logger.debug("Worker thread %s finished", worker_id)


def _emit_error_result(
    output_queue: Queue,
    e: Exception,
    request_config: Optional[Any],
    worker_id: int,
) -> None:
    """Emit an error RequestMetrics tuple to the output queue.

    Mirrors the standard error path to keep counters consistent across
    all failure scenarios.
    """
    try:
        prompt_len = request_config.prompt[1] if request_config is not None else 0

        error_code = None
        # Check for HTTP error codes from requests library
        if hasattr(e, "response") and hasattr(e.response, "status_code"):
            error_code = e.response.status_code

        metrics = RequestMetrics(
            request_dispatched_at=0.0,
            inter_token_times=[],
            num_prompt_tokens=prompt_len,
            num_output_tokens=0,
            error_msg=str(e),
            error_code=error_code,
            request_id=request_config.id if request_config else None,
        )
        output_queue.put((metrics, None))
    except Exception:
        logger.exception(
            "Failed to enqueue error result for worker %s",
            worker_id,
        )


class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs using a thread pool."""

    def __init__(
        self,
        client_config: ClientConfig,
        input_queue: Queue,
        output_queue: Queue,
    ):
        self.client_config = client_config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.workers: List[Thread] = []

    def start(self) -> None:
        """Start the configured number of worker threads."""
        logger.info(f"Starting {self.client_config.num_threads} worker threads")
        for i in range(self.client_config.num_threads):
            worker = Thread(
                target=_worker_thread,
                args=(self.input_queue, self.output_queue, i, self.client_config),
                name=f"worker-{i}",
                daemon=False,
            )
            self.workers.append(worker)
            worker.start()

    def get_worker_count(self) -> int:
        """Return the number of worker threads."""
        return len(self.workers)

    def complete_tasks(self) -> None:
        """Signal worker threads to complete their tasks and exit."""
        # Send one sentinel value per worker
        for _ in range(len(self.workers)):
            self.input_queue.put(None)

    def wait_for_workers(self) -> None:
        """Wait for all worker threads to complete their tasks and exit."""
        self.complete_tasks()
        for worker in self.workers:
            worker.join()
