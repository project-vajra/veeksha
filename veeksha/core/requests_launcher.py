import asyncio
import functools
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from queue import Empty
from typing import List

import aiohttp
from veeksha.config.client import ClientConfig
from veeksha.core.llm_clients import construct_client
from veeksha.core.llm_clients.base_llm_client import BaseLLMClient
from veeksha.logger import init_logger

logger = init_logger(__name__)


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

    def start(self) -> None:
        """Start the clients."""
        for client_id in range(self.client_config.num_clients):
            client = Process(
                target=self.run_client,
                args=(client_id,),
            )
            self.clients.append(client)
            client.start()

    def run_client(self, client_id: int) -> None:
        """Run a client process that sends requests to the LLM API."""
        try:
            asyncio.run(self.run_async_worker(client_id))
        except KeyboardInterrupt:
            logger.info("Worker %s received KeyboardInterrupt", client_id)

    async def run_async_worker(self, client_id: int) -> None:
        """Run the async worker that processes requests."""
        logger.info("Starting async worker %s", client_id)

        # Create LLM client for this worker process
        llm_client = construct_client(
            model_name=self.client_config.model,
            tokenizer_name=self.client_config.tokenizer,
            llm_api=self.client_config.llm_api,
        )

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.client_config.request_timeout)
        ) as session:
            tasks = [
                asyncio.create_task(
                    self.process_requests_async(client_id, i, llm_client, session)
                )
                for i in range(self.client_config.num_concurrent_requests_per_client)
            ]
            await asyncio.gather(*tasks)

        logger.info("Async worker %s finished", client_id)

    async def process_requests_async(
        self,
        client_id: int,
        task_id: int,
        llm_client: BaseLLMClient,
        session: aiohttp.ClientSession,
    ) -> None:
        """A task that processes requests from the input queue."""
        loop = asyncio.get_running_loop()
        get_from_queue = functools.partial(self.input_queue.get, timeout=1.0)
        put_to_queue = functools.partial(self.output_queue.put)

        while True:
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
                except Exception:
                    logger.exception(
                        "send_llm_request failed for client_id=%s, task_id=%s",
                        client_id,
                        task_id,
                    )
                    continue

            except asyncio.CancelledError:
                logger.warning(
                    "Worker %s task %s cancelled during queue operation",
                    client_id,
                    task_id,
                )
                break
            except Exception:
                logger.exception(
                    "Unexpected error in worker %s task %s", client_id, task_id
                )
                break

    def complete_tasks(self) -> None:
        """Signal worker processes to complete their tasks and exit."""
        for _ in range(
            self.client_config.num_clients
            * self.client_config.num_concurrent_requests_per_client
        ):
            self.input_queue.put(None)

    def kill_clients(self) -> None:
        """Wait for all clients to complete their tasks and exit."""
        self.complete_tasks()
        for client in self.clients:
            client.join()