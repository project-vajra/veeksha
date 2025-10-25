import asyncio
import functools
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from queue import Empty
from threading import Lock
from typing import Any, Callable, List, Optional

import aiohttp  # type: ignore

from veeksha.config.client import ClientConfig
from veeksha.core.llm_clients import construct_client
from veeksha.core.llm_clients.base_llm_client import BaseLLMClient
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


def _client_process_entry(
    client_config: ClientConfig,
    input_queue: MPQueue,
    output_queue: MPQueue,
    client_id: int,
) -> None:
    """Module-level entry point for worker processes.

    Using a top-level function avoids pickling the RequestsLauncher instance on
    platforms that use the "spawn" method (e.g., Windows).
    """
    asyncio.run(
        _run_async_worker(
            client_config=client_config,
            input_queue=input_queue,
            output_queue=output_queue,
            client_id=client_id,
        )
    )


async def _run_async_worker(
    client_config: ClientConfig,
    input_queue: MPQueue,
    output_queue: MPQueue,
    client_id: int,
) -> None:
    """Run the async worker that processes requests for a single process."""
    logger.debug("Starting async worker %s", client_id)

    llm_client = construct_client(
        model_name=client_config.model,
        tokenizer_name=client_config.tokenizer or client_config.model,
        llm_api=client_config.llm_api,
    )

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=client_config.request_timeout)
    ) as session:
        tasks = [
            asyncio.create_task(
                _process_requests_async(
                    input_queue=input_queue,
                    output_queue=output_queue,
                    client_id=client_id,
                    task_id=i,
                    llm_client=llm_client,
                    session=session,
                )
            )
            for i in range(client_config.num_concurrent_requests_per_client)
        ]
        await asyncio.gather(*tasks)

    logger.debug("Async worker %s finished", client_id)


async def _process_requests_async(
    input_queue: MPQueue,
    output_queue: MPQueue,
    client_id: int,
    task_id: int,
    llm_client: "BaseLLMClient",
    session: aiohttp.ClientSession,
) -> None:
    """A task that processes requests from the input queue."""
    loop = asyncio.get_running_loop()
    get_from_queue = functools.partial(input_queue.get, timeout=1.0)
    put_to_queue = functools.partial(output_queue.put)

    while True:
        request_config = None
        try:
            try:
                request_config = await loop.run_in_executor(None, get_from_queue)
            except Empty:
                continue  # Poll the queue again

            if request_config is None:
                break

            try:
                result = await llm_client.send_llm_request(request_config, session)
                await loop.run_in_executor(None, put_to_queue, result)
            except asyncio.CancelledError:
                logger.debug("Worker %s task %s was cancelled", client_id, task_id)
                break
            except Exception as e:
                logger.exception(
                    "send_llm_request failed for client_id=%s, task_id=%s",
                    client_id,
                    task_id,
                )
                await _emit_error_result(
                    loop=loop,
                    put_to_queue=put_to_queue,
                    e=e,
                    request_config=request_config,
                    client_id=client_id,
                    task_id=task_id,
                )
                continue

        except asyncio.CancelledError:
            logger.warning(
                "Worker %s task %s cancelled during queue operation",
                client_id,
                task_id,
            )
            break
        except Exception as e:
            logger.exception(
                "Unexpected error in worker %s task %s", client_id, task_id
            )
            await _emit_error_result(
                loop=loop,
                put_to_queue=put_to_queue,
                e=e,
                request_config=request_config,
                client_id=client_id,
                task_id=task_id,
            )
            continue


async def _emit_error_result(
    loop: asyncio.AbstractEventLoop,
    put_to_queue: "Callable[[Any], None]",
    e: Exception,
    request_config: Optional[Any],
    client_id: int,
    task_id: int,
) -> None:
    """Emit an error RequestMetrics tuple to the output queue.

    Mirrors the standard error path to keep counters consistent across
    all failure scenarios.
    """
    try:
        prompt_len = request_config.prompt[1] if request_config is not None else 0

        error_code = None
        if isinstance(e, aiohttp.ClientResponseError):
            error_code = e.status  # type: ignore[attr-defined]
        metrics = RequestMetrics(
            request_dispatched_at=0.0,
            inter_token_times=[],
            num_prompt_tokens=prompt_len,
            num_output_tokens=0,
            error_msg=str(e),
            error_code=error_code,
            request_id=request_config.id if request_config else None,
        )
        await loop.run_in_executor(None, put_to_queue, (metrics, None))
    except Exception:
        logger.exception(
            "Failed to enqueue error result for worker %s task %s",
            client_id,
            task_id,
        )


class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(
        self,
        client_config: ClientConfig,
        input_queue: MPQueue,
        output_queue: MPQueue,
    ):
        self.client_config = client_config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.clients: List[Process] = []
        self._lock: Lock = Lock()
        self._next_client_id: int = 0

    def start(self) -> None:
        """Start the initial configured clients."""
        with self._lock:
            for _ in range(self.client_config.num_clients):
                if self._can_spawn_more_locked():
                    self._spawn_client_locked()
                else:
                    logger.warning(
                        "Configured num_clients=%d exceeds max_clients=%s; starting only %d",
                        self.client_config.num_clients,
                        str(self.client_config.max_clients),
                        len(self.clients),
                    )
                    break

    def _spawn_client_locked(self) -> None:
        client_id = self._next_client_id
        client = Process(
            target=_client_process_entry,
            args=(self.client_config, self.input_queue, self.output_queue, client_id),
        )
        self.clients.append(client)
        self._next_client_id += 1
        client.start()

    def spawn_new_client(self) -> None:
        """Dynamically spawn a new client process that persists for the run."""
        with self._lock:
            if self._can_spawn_more_locked():
                self._spawn_client_locked()
            else:
                logger.info(
                    "Max clients reached (max_clients=%s); not spawning new client.",
                    str(self.client_config.max_clients),
                )

    def get_client_count(self) -> int:
        with self._lock:
            return len(self.clients)

    def get_total_slots(self) -> int:
        with self._lock:
            return (
                len(self.clients)
                * self.client_config.num_concurrent_requests_per_client
            )

    def can_spawn_more(self) -> bool:
        with self._lock:
            return self._can_spawn_more_locked()

    def _can_spawn_more_locked(self) -> bool:
        max_clients = self.client_config.max_clients
        return max_clients is None or len(self.clients) < max_clients

    def complete_tasks(self) -> None:
        """Signal worker processes to complete their tasks and exit."""
        total_slots = self.get_total_slots()
        for _ in range(total_slots):
            self.input_queue.put(None)

    def wait_for_clients(self, timeout: float = 2.0) -> None:
        """Wait for all clients to complete their tasks and exit.

        Args:
            timeout: Maximum time to wait for each client to join (seconds)
        """
        self.complete_tasks()

        for i, client in enumerate(self.clients):
            logger.info(f"Waiting for client {i} to join (timeout={timeout}s)...")
            client.join(timeout=timeout)
            if client.is_alive():
                logger.error(
                    f"Client {i} (PID={client.pid}) did not exit within {timeout}s, "
                    "terminating forcefully"
                )
                client.terminate()
                client.join(timeout=5.0)
                if client.is_alive():
                    logger.error(f"Client {i} still alive after terminate, killing")
                    client.kill()
                    client.join()
            else:
                logger.info(f"Client {i} joined successfully")
