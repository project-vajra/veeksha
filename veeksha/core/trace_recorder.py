import json
import queue
import threading
import time
from typing import Any, Dict, List, Optional

from veeksha.config.trace_recorder import TraceRecorderConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


class TraceRecorder:
    """Records dispatched requests to a JSONL trace file asynchronously."""

    def __init__(
        self,
        output_dir: str,
        benchmark_start_time: float,
        config: TraceRecorderConfig,
    ):
        """Initialize the trace recorder.

        Args:
            config: Trace recorder configuration.
        """
        self.config = config
        self.output_dir = output_dir
        self.benchmark_start_time = benchmark_start_time
        self.include_content = self.config.include_content
        self.trace_file_path = f"{self.output_dir}/dispatch_trace.jsonl"

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        self._batch_size = 100
        self._flush_interval = 1.0  # seconds

        try:
            with open(self.trace_file_path, "w") as f:
                pass
            logger.info(f"Initialized trace file at {self.trace_file_path}")
        except Exception as e:
            logger.error(f"Failed to initialize trace file: {e}")

    def start(self) -> None:
        """Start the background writer thread."""
        if self._writer_thread is not None:
            return

        self._stop_event.clear()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="TraceRecorderWriter", daemon=True
        )
        self._writer_thread.start()
        logger.info("Trace recorder writer thread started")

    def stop(self) -> None:
        """Stop the writer thread and flush remaining items."""
        if self._writer_thread is None:
            return

        logger.info("Stopping trace recorder, flushing remaining items...")
        self._stop_event.set()
        self._writer_thread.join(timeout=5.0)
        self._writer_thread = None

    def record_dispatch(
        self,
        request: Any,
        session_id: int,
        session_size: int,
        dispatched_at: float,
    ) -> None:
        """Queue a dispatched request for recording.

        Args:
            request: The dispatched request object
            session_id: Session ID
            session_size: Total requests in the session
            dispatched_at: Monotonic timestamp of dispatch
        """
        try:
            channels_data = None
            history_data = None

            trace_entry = {
                "request_id": request.id,
                "session_id": session_id,
                "session_size": session_size,
                "dispatched_at": round(dispatched_at - self.benchmark_start_time, 5),
                "session_context": request.session_context,
            }

            if self.include_content:
                channels_data = {
                    str(modality.name).lower(): self._serialize_channel_content(content)
                    for modality, content in request.channels.items()
                }
                trace_entry["channels"] = channels_data
                trace_entry["history"] = request.history

            self._queue.put(trace_entry)

        except Exception as e:
            logger.error(f"Failed to queue trace for request {request.id}: {e}")

    def _writer_loop(self) -> None:
        """Background loop to write traces to file."""
        buffer: List[Dict[str, Any]] = []
        last_flush_time = time.monotonic()

        with open(self.trace_file_path, "a") as f:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    timeout = 0.1
                    try:
                        item = self._queue.get(timeout=timeout)
                        buffer.append(item)
                    except queue.Empty:
                        pass

                    current_time = time.monotonic()
                    should_flush = (
                        len(buffer) >= self._batch_size
                        or (
                            buffer
                            and (current_time - last_flush_time) >= self._flush_interval
                        )
                        or (self._stop_event.is_set() and buffer)
                    )

                    if should_flush:
                        self._flush_buffer(f, buffer)
                        buffer.clear()
                        last_flush_time = current_time

                except Exception as e:
                    logger.error(f"Error in trace writer loop: {e}")
                    buffer.clear()

    def _flush_buffer(self, f, buffer: List[Dict[str, Any]]) -> None:
        """Write buffered items to file."""
        try:
            lines = [json.dumps(entry) + "\n" for entry in buffer]
            f.writelines(lines)
            f.flush()
        except Exception as e:
            logger.error(f"Failed to write batch of {len(buffer)} traces: {e}")

    def _serialize_channel_content(self, content: Any) -> Dict[str, Any]:
        """Serialize channel content to a dictionary."""
        if hasattr(content, "__dataclass_fields__"):
            from dataclasses import asdict

            return asdict(content)
        try:
            return vars(content)
        except TypeError:
            return {"raw_str": str(content)}
