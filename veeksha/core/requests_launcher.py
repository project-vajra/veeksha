import asyncio
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from threading import Thread

import aiohttp

from veeksha.config.client import ClientConfig
from veeksha.core.llm_clients import construct_client
from veeksha.core.response import Response
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


def _process_requests(
    llm_client,
    input_queue: MPQueue,
    output_queue: MPQueue,
) -> None:
    """Process requests from the input queue and send them to the LLM API."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def main():
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            while True:
                request_config = input_queue.get()
                if request_config is None:
                    break

                try:
                    result = await llm_client.send_llm_request(request_config, session)
                    output_queue.put(result)
                except Exception as e:
                    logger.exception("send_llm_request failed")
                    prompt_len = (
                        request_config.prompt[1]
                        if request_config.prompt and len(request_config.prompt) > 1
                        else 0
                    )
                    request_metrics = RequestMetrics(
                        request_id=request_config.request_id, prompt_len=prompt_len
                    )
                    error_response = Response(
                        request_id=request_config.request_id,
                        error=str(e),
                        prompt_len=prompt_len,
                    )
                    output_queue.put((request_metrics, error_response))
                    continue

    try:
        loop.run_until_complete(main())
    finally:
        try:
            asyncio.set_event_loop(None)
        finally:
            loop.close()


def _run_client(
    client_config: ClientConfig,
    input_queue: MPQueue,
    output_queue: MPQueue,
) -> None:
    """Run a client process that sends requests to the LLM API."""
    llm_client = construct_client(
        model_name=client_config.model,
        tokenizer_name=client_config.tokenizer,
        llm_api=client_config.llm_api,
    )

    threads = [
        Thread(target=_process_requests, args=(llm_client, input_queue, output_queue))
        for _ in range(client_config.num_concurrent_requests_per_client)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


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

        self.clients = [
            Process(
                target=_run_client,
                args=(
                    self.client_config,
                    self.input_queue,
                    self.output_queue,
                ),
            )
            for _ in range(self.client_config.num_clients)
        ]

    def start(self) -> None:
        """Start the clients."""
        for client in self.clients:
            client.start()

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
