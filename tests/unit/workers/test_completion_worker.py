from __future__ import annotations

import threading
from queue import Queue
from unittest.mock import MagicMock

import pytest

from veeksha.core.context import WorkerContext
from veeksha.core.response import RequestResult
from veeksha.workers.completion import CompletionWorker


@pytest.mark.unit
def test_completion_worker_processes_results_after_shared_stop() -> None:
    output_queue = Queue()
    traffic_scheduler = MagicMock()
    evaluator = MagicMock()
    stop_event = threading.Event()
    stop_event.set()
    worker = CompletionWorker(
        output_queue=output_queue,
        traffic_scheduler=traffic_scheduler,
        evaluator=evaluator,
        worker_context=WorkerContext(worker_id=0, stop_event=stop_event),
    )
    thread = threading.Thread(target=worker.run)
    result = RequestResult(
        request_id=7,
        session_id=3,
        channels={},
        success=True,
        client_completed_at=123.0,
    )

    thread.start()
    thread.join(timeout=0.1)
    assert thread.is_alive()

    output_queue.put(result)
    output_queue.put(None)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    traffic_scheduler.notify_completion.assert_called_once_with(
        request_id=7,
        completed_at_monotonic=123.0,
        success=True,
        channel_responses={},
    )
    evaluator.record_request_completed.assert_called_once_with(
        request_id=7,
        session_id=3,
        completed_at=123.0,
        response=result,
        error=None,
    )
