import asyncio
import time
from multiprocessing import Process, Manager
from multiprocessing import Queue as MPQueue
from typing import Dict, List, Tuple

from veeksha.core.response import Response
from veeksha.config.config import ClientConfig
from veeksha.core.llm_clients import construct_client
from veeksha.core.llm_clients.base_llm_client import BaseLLMClient
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)

class RequestsLauncher:
    """Launch requests from LLMClients to their respective LLM APIs."""

    def __init__(
        self,
        client_config: ClientConfig,
        input_queue: MPQueue,
        output_queue: MPQueue,
        total_benchmark_time: float,
    ):
        self.clients = []
        self.llm_clients: Dict[int, BaseLLMClient] = {}
        
        # Create a shared manager for the request_metrics dictionary
        self.manager = Manager()
        self.request_outputs = self.manager.dict() # {client_id: {request_id: [metrics, output, is_finished]}}
        
        self.client_config = client_config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.total_benchmark_time = total_benchmark_time

        # Initialize the nested structure for each client
        for client_id in range(self.client_config.num_clients):
            self.request_outputs[client_id] = self.manager.dict()
            client = Process(
                target=self.run_client,
                args=(client_id,),
            )
            self.clients.append(client)

    def start(self) -> None:
        """Start the clients."""
        for client in self.clients:
            client.start()

    def run_client(self, client_id: int) -> None:
        """Run the client."""
        assert self.client_config.tokenizer is not None
        assert self.client_config.model is not None

        self.llm_clients[client_id] = construct_client(
            model_name=self.client_config.model,
            tokenizer_name=self.client_config.tokenizer,
            llm_api=self.client_config.llm_api,
        )
        # Dictionary already initialized in __init__
        self.start_threads(client_id=client_id)

    def start_threads(self, client_id: int) -> None:
        """Start the asyncio tasks."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        tasks = [
            self.process_requests(client_id)
            for _ in range(self.client_config.num_concurrent_requests_per_client)
        ]
        
        try:
            loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            loop.close()

    async def process_requests(self, client_id: int) -> None:
        while True:
            # Use an executor to perform blocking queue operations
            request_config = await asyncio.to_thread(self.input_queue.get)
            if request_config is None:
                break
            logger.info("-" * 80)
            logger.info(f"Client {client_id} sent request {request_config.metadata['request_id']}")
            logger.info(f"Number of prefill tokens: {request_config.metadata['num_prefill_tokens']}")
            logger.info(f"Session id: {request_config.metadata['session_id']}")
            logger.info(f"Number of requests in session: {request_config.metadata['num_requests_in_session']}")
            
            request_dispatched_at = time.monotonic() - self.llm_clients[client_id].start_time

            # add to data structure
            request_id = request_config.metadata['request_id']
            client_requests = dict(self.request_outputs[client_id])

            client_requests[request_id] = [RequestMetrics(
                request_dispatched_at=request_dispatched_at,
                inter_token_times=[],
                num_prompt_tokens=request_config.metadata['num_prefill_tokens'],
                num_output_tokens=0,
                error_code=-1,
                error_msg="Request not finished",
            ), Response(
            id=request_config.id,
            text="",
            ), False]

            self.request_outputs[client_id] = client_requests

            # will update metrics with response stream
            async def token_callback(req_config, token_times_to_append):
                await self.on_token_received(client_id, req_config, token_times_to_append)
                
            metrics, response = await self.llm_clients[client_id].send_llm_request(
                request_config, 
                request_dispatched_at,
                on_token_callback=token_callback, # because we want to also track unfinished requests
            )

            # update data structure: request has completed
            client_requests = dict(self.request_outputs[client_id])
            client_requests[request_id] = [metrics, response, True]
            self.request_outputs[client_id] = client_requests
            
            # Use an executor to put result in queue
            await asyncio.to_thread(self.output_queue.put, (metrics, response))

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

    async def on_token_received(self, client_id, request_config, token_times_to_append):
        """Update request metrics and response in real-time as tokens arrive.
        
        Args:
            client_id: The ID of the client processing the request
            request_config: The request configuration
            token_times_to_append: New token timing data to append to metrics
        
        This method is designed to be thread-safe when used with multiprocessing.Manager dictionaries.
        """
        try:
            if hasattr(request_config, 'metadata') and 'request_id' in request_config.metadata:
                request_id = request_config.metadata['request_id']
                
                # Safe check if the dictionary keys exist
                if client_id in self.request_outputs and request_id in self.request_outputs[client_id]:
                    # Make a complete copy of the nested dictionary to avoid race conditions
                    client_requests = dict(self.request_outputs[client_id])
                    current_entry = client_requests[request_id]
                    
                    # Extract current metrics and completion status
                    current_metrics = current_entry[0]
                    current_response = current_entry[1]
                    is_completed = current_entry[2] if len(current_entry) > 2 else False
                    
                    # Update metrics atomically
                    current_metrics.num_output_tokens += len(token_times_to_append)
                    current_metrics.inter_token_times.extend(token_times_to_append)
                    
                    # Create updated entry with the same structure
                    client_requests[request_id] = [
                        current_metrics,  # Updated metrics
                        current_response,  # Keep current response
                        is_completed      # Preserve completion status
                    ]
                    
                    # Atomic update of the entire dictionary at once
                    self.request_outputs[client_id] = client_requests
        except Exception as e:
            logger.error(f"Error in on_token_received: {e}")
            # Don't re-raise - we want to continue processing even if updates fail

    def kill_clients(self) -> None:
        """Kill all the clients."""
        for client in self.clients:
            client.terminate()
            client.join(30)
            client.kill()
            client.close()
            
    def get_request_outputs(self) -> Dict:
        """Get the request outputs data structure.
        
        Returns:
            Dict mapping client_id to a dictionary of request_id -> [RequestMetrics, Response, is_finished]
        """
        # Convert manager dictionary to regular Python dictionary for easier debugging
        result = {}
        for client_id in self.request_outputs.keys():
            result[client_id] = dict(self.request_outputs[client_id])
            
        return result

    def get_finished_requests_count(self) -> int:
        """Get the number of finished requests.
        
        Returns:
            The count of requests that have been marked as finished.
        """
        count = 0
        for client_id, request_metrics in self.get_request_outputs().items():
            for request_id, (metrics, response, finished) in request_metrics.items():
                if finished:
                    count += 1
        return count
        
    def get_unfinished_requests_count(self) -> int:
        """Get the number of unfinished requests.
        
        Returns:
            The count of requests that are still in progress.
        """
        count = 0
        metrics_dict = self.get_request_outputs()
        
        for client_id, request_metrics in metrics_dict.items():
            for request_id, (metrics, response, finished) in request_metrics.items():
                if not finished:
                    count += 1
        return count

    def get_unfinished_requests(self) -> Dict[int, List[Tuple[RequestMetrics, Response]]]:
        """Get the unfinished requests.
        
        Returns:
            Dict mapping client_id to List of (RequestMetrics, Response) for unfinished requests
        """
        metrics_dict = self.get_request_outputs()
        unfinished_requests = {client_id: [] for client_id in metrics_dict}
        
        for client_id, request_metrics in metrics_dict.items():
            for request_id, (metrics, response, finished) in request_metrics.items():
                if not finished:
                    unfinished_requests[client_id].append((metrics, response))
        
        return unfinished_requests
