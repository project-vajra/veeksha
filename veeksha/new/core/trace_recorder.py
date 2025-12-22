"""Trace recorder for logging dispatched requests for replay."""

import json
import threading
from typing import Any, Dict

from veeksha.logger import init_logger
from veeksha.new.config.trace_recorder import TraceRecorderConfig

logger = init_logger(__name__)


class TraceRecorder:
    """Records dispatched requests to a JSONL trace file."""

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
        self._lock = threading.Lock()

        try:
            with open(self.trace_file_path, "w") as f:
                pass
            logger.info(f"Initialized trace file at {self.trace_file_path}")
        except Exception as e:
            logger.error(f"Failed to initialize trace file: {e}")

    def record_dispatch(
        self,
        request: Any,
        session_id: int,
        session_size: int,
        dispatched_at: float,
    ) -> None:
        """Record a dispatched request to the trace file.

        Args:
            request: The dispatched request object
            session_id: Session ID
            session_size: Total requests in the session
            dispatched_at: Monotonic timestamp of dispatch
        """
        # Serialize channels
        channels_data = None
        history_data = None

        if self.include_content:
            channels_data = {
                str(modality.name).lower(): self._serialize_channel_content(content)
                for modality, content in request.channels.items()
            }
            history_data = request.history

        trace_entry = {
            "request_id": request.id,
            "session_id": session_id,
            "session_size": session_size,
            "dispatched_at": round(dispatched_at - self.benchmark_start_time, 5),
            "channels": channels_data,
            "history": history_data,
            "session_context": request.session_context,
        }

        try:
            json_line = json.dumps(trace_entry)
            with self._lock:
                with open(self.trace_file_path, "a") as f:
                    f.write(json_line + "\n")
        except Exception as e:
            logger.error(f"Failed to record trace for request {request.id}: {e}")

    def _serialize_channel_content(self, content: Any) -> Dict[str, Any]:
        """Serialize channel content to a dictionary."""
        if hasattr(content, "__dataclass_fields__"):
            from dataclasses import asdict

            return asdict(content)
        try:
            return vars(content)
        except TypeError:
            return {"raw_str": str(content)}
