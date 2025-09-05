import asyncio
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from typing import Dict

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
        self.clients = []
        self.llm_clients: Dict[int, BaseLLMClient] = {}

        self.client_config = client_config
        self.input_queue = input_queue
        self.output_queue = output_queue

        for client_id in range(self.client_config.num_clients):
            client = Process(
                target=self.run_async_worker,
                args=(client_id,),
            )
            self.clients.append(client)

    def start(self) -> None:
        """Start the clients."""
        for client in self.clients:
            client.start()

    def run_async_worker(self, client_id: int) -> None:
        """Run an async worker process."""
        try:
            # Each process runs its own asyncio event loop
            asyncio.run(self.async_worker_main(client_id))
        except Exception:
            logger.exception("Async worker %s crashed", client_id)

    async def async_worker_main(self, client_id: int) -> None:
        """Main async function for each worker process."""
        assert self.client_config.tokenizer is not None
        assert self.client_config.model is not None

        # Create LLM client for this worker process
        llm_client = construct_client(
            model_name=self.client_config.model,
            tokenizer_name=self.client_config.tokenizer,
            llm_api=self.client_config.llm_api,
        )

        # Create a single aiohttp session for this worker process
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            # Create multiple concurrent tasks to handle requests
            tasks = []
            for task_id in range(self.client_config.num_concurrent_requests_per_client):
                task = asyncio.create_task(
                    self.process_requests_async(client_id, task_id, llm_client, session)
                )
                tasks.append(task)

            # Wait for all tasks to complete
            try:
                await asyncio.gather(*tasks)
            except Exception:
                logger.exception("Error in async worker %s tasks", client_id)
                # Cancel any remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Wait for cancellation to complete
                await asyncio.gather(*tasks, return_exceptions=True)

    async def process_requests_async(
        self, client_id: int, task_id: int, llm_client: BaseLLMClient, session: aiohttp.ClientSession
    ) -> None:
        """Process requests asynchronously within a worker process."""
        logger.debug("Starting async task %s for worker %s", task_id, client_id)
        
        while True:
            try:
                # Get request from multiprocessing queue (this is a blocking operation)
                # We need to run it in a thread executor to avoid blocking the event loop
                request_config = await asyncio.get_event_loop().run_in_executor(
                    None, self.input_queue.get
                )
                
                if request_config is None:
                    logger.debug("Worker %s task %s received shutdown signal", client_id, task_id)
                    break

                # Process the request asynchronously
                try:
                    result = await llm_client.send_llm_request(request_config, session)
                    
                    # Put result in output queue (also run in executor to avoid blocking)
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.output_queue.put, result
                    )
                except asyncio.CancelledError:
                    logger.debug("Worker %s task %s was cancelled", client_id, task_id)
                    raise
                except Exception:
                    logger.exception(
                        "send_llm_request failed for worker %s task %s", client_id, task_id
                    )
                    continue

            except asyncio.CancelledError:
                logger.debug("Worker %s task %s cancelled during queue operation", client_id, task_id)
                break
            except Exception:
                logger.exception("Unexpected error in worker %s task %s", client_id, task_id)
                break
        
        logger.debug("Async task %s for worker %s completed", task_id, client_id)

    def complete_tasks(self) -> None:
        """Complete the clients."""
        # put None to indicate that client should stop
        for _ in range(
            self.client_config.num_clients
            * self.client_config.num_concurrent_requests_per_client
        ):
            self.input_queue.put(None)

        for client in self.clients:
            client.join()

    def kill_clients(self) -> None:
        """Kill all the clients."""
        for client in self.clients:
            client.terminate()
            client.join(30)
            client.kill()
            client.close()