"""Performance evaluator for TTS audio generation metrics."""

import json
import os
import struct
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from veeksha.config.evaluator import (
    AudioChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.evaluator.base import EvaluationResult
from veeksha.evaluator.cdf_sketch import CDFSketch
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

logger = init_logger(__name__)

SAMPLE_RATE_DEFAULT = 24000


def _make_wav_header(data_size: int, sample_rate: int = SAMPLE_RATE_DEFAULT) -> bytes:
    """Build a 44-byte WAV header for raw PCM data."""
    byte_rate = sample_rate * 1 * 16 // 8
    block_align = 1 * 16 // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        1,  # mono
        sample_rate,
        byte_rate,
        block_align,
        16,  # bits per sample
        b"data",
        data_size,
    )


@dataclass
class AudioRequestMetrics:
    """Metrics for a single audio (TTS) request."""

    request_id: int
    session_id: int
    request_dispatched_at: float
    client_completed_at: float
    ttfa: float
    end_to_end_latency: float
    generated_audio_duration: float
    rtf: float
    chunk_count: int
    pcm_byte_count: int
    input_chars: int = 0
    input_tokens: int = 0
    session_total_requests: Optional[int] = None


class AudioPerformanceEvaluator:
    """Performance evaluator for audio generation (TTS).

    Tracks per-request TTFA, total latency, audio duration, RTF (real-time factor),
    and chunk count. Computes p50/p90/p99 aggregates via CDFSketch.
    """

    def __init__(
        self,
        config: PerformanceEvaluatorConfig,
        channel_config: Optional[AudioChannelPerformanceConfig] = None,
        benchmark_start_time: float = 0.0,
    ):
        self.config = config
        self.channel_config = channel_config or AudioChannelPerformanceConfig()
        self.benchmark_start_time = benchmark_start_time
        self.lock = threading.Lock()

        # CDF sketches for aggregate metrics
        self.summaries: Dict[str, CDFSketch] = {
            "ttfa": CDFSketch("ttfa", unit="ms"),
            "end_to_end_latency": CDFSketch("end_to_end_latency", unit="ms"),
            "generated_audio_duration": CDFSketch(
                "generated_audio_duration", unit="ms"
            ),
            "rtf": CDFSketch("rtf"),
            "chunk_count": CDFSketch("chunk_count"),
            "input_tokens": CDFSketch("input_tokens", unit="tokens"),
            "session_size": CDFSketch("session_size"),
            "session_duration": CDFSketch("session_duration", unit="ms"),
        }

        # Request dispatch tracking (like text.py _pending_requests)
        self._pending_requests: Dict[int, Dict[str, Any]] = {}

        # Request-level storage for JSONL output
        self._completed_metrics: List[AudioRequestMetrics] = []
        self._lifecycle_timestamps: List[Dict[str, Optional[float]]] = []
        self._request_rows_streamed: int = 0
        self._request_time_reference: float = self.benchmark_start_time

        # Audio bytes storage for optional WAV saving
        self._audio_buffers: Dict[int, bytes] = {}
        self._audio_metadata: Dict[int, Dict[str, Any]] = {}

        # Running totals for aggregate throughput
        self._total_input_chars: int = 0
        self._first_dispatch_at: Optional[float] = None
        self._last_completion_at: Optional[float] = None

    def register_request(
        self,
        request_id: int,
        session_id: int,
        dispatched_at: float,
        content: Any,
        requested_output: Any = None,
    ) -> None:
        """Register an audio request that was dispatched."""
        with self.lock:
            if self._request_time_reference == 0.0:
                self._request_time_reference = dispatched_at

            self._pending_requests[request_id] = {
                "session_id": session_id,
                "dispatched_at": dispatched_at,
            }

    def record_request_completed(
        self,
        request_id: int,
        session_id: int,
        completed_at: float,
        response: Any,
    ) -> None:
        """Record that an audio request completed."""
        with self.lock:
            dispatch_info = self._pending_requests.pop(request_id, None)
            if dispatch_info:
                dispatched_at = dispatch_info["dispatched_at"]
            else:
                dispatched_at = getattr(
                    response,
                    "scheduler_dispatched_at",
                    self._request_time_reference,
                )

            channel_response = None
            if hasattr(response, "channels"):
                channel_response = response.channels.get(ChannelModality.AUDIO)

            if channel_response is None:
                logger.debug("Request %d has no AUDIO channel response", request_id)
                return

            cm = channel_response.metrics or {}
            ttfa = cm.get("ttfa", 0.0)
            end_to_end_latency = cm.get("end_to_end_latency", 0.0)
            generated_audio_duration = cm.get("generated_audio_duration", 0.0)
            rtf = cm.get("rtf", 0.0)
            chunk_count = cm.get("chunk_count", 0)
            pcm_byte_count = cm.get("pcm_byte_count", 0)
            input_chars = cm.get("input_chars", 0)
            input_tokens = cm.get("input_tokens", 0)

            session_total_requests = getattr(response, "session_total_requests", None)

            metrics = AudioRequestMetrics(
                request_id=request_id,
                session_id=session_id,
                request_dispatched_at=dispatched_at,
                client_completed_at=completed_at,
                ttfa=ttfa,
                end_to_end_latency=end_to_end_latency,
                generated_audio_duration=generated_audio_duration,
                rtf=rtf,
                chunk_count=chunk_count,
                pcm_byte_count=pcm_byte_count,
                input_chars=input_chars,
                input_tokens=input_tokens,
                session_total_requests=session_total_requests,
            )

            self._completed_metrics.append(metrics)

            # Update aggregate throughput accumulators
            self._total_input_chars += input_chars
            if self._first_dispatch_at is None or dispatched_at < self._first_dispatch_at:
                self._first_dispatch_at = dispatched_at
            if self._last_completion_at is None or completed_at > self._last_completion_at:
                self._last_completion_at = completed_at

            # Store lifecycle timestamps (matching text.py pattern)
            def normalize_ts(ts: Optional[float]) -> Optional[float]:
                if ts is None:
                    return None
                return round(max(0.0, ts - self._request_time_reference), 5)

            self._lifecycle_timestamps.append(
                {
                    "scheduler_ready_at": normalize_ts(
                        getattr(response, "scheduler_ready_at", None)
                    ),
                    "scheduler_dispatched_at": normalize_ts(
                        getattr(response, "scheduler_dispatched_at", None)
                    ),
                    "client_picked_up_at": normalize_ts(
                        getattr(response, "client_picked_up_at", None)
                    ),
                    "client_completed_at": normalize_ts(
                        getattr(response, "client_completed_at", None)
                    ),
                    "result_processed_at": normalize_ts(
                        getattr(response, "result_processed_at", None)
                    ),
                }
            )

            # Update CDF sketches
            self.summaries["ttfa"].put(ttfa)
            self.summaries["end_to_end_latency"].put(end_to_end_latency)
            self.summaries["generated_audio_duration"].put(generated_audio_duration)
            self.summaries["rtf"].put(rtf)
            self.summaries["chunk_count"].put(chunk_count)
            self.summaries["input_tokens"].put(input_tokens)

            # Store audio buffer for optional WAV saving
            if self.channel_config.save_audio_files:
                audio_content = channel_response.content
                if audio_content and isinstance(audio_content, bytes):
                    self._audio_buffers[request_id] = audio_content
                    self._audio_metadata[request_id] = {
                        "raw_pcm": cm.get("raw_pcm", False),
                        "sample_rate": cm.get("sample_rate", SAMPLE_RATE_DEFAULT),
                    }

    def record_session_completed(
        self,
        session_id: int,
        session_size: int,
        first_dispatch_at: Optional[float],
        last_completion_at: Optional[float],
    ) -> None:
        """Record session-level metrics."""
        with self.lock:
            self.summaries["session_size"].put(session_size)

            if first_dispatch_at is not None and last_completion_at is not None:
                duration = max(0.0, last_completion_at - first_dispatch_at)
                self.summaries["session_duration"].put(duration)

    def get_summary(self) -> Dict[str, Optional[float]]:
        """Get summary metrics from all CDF sketches."""
        perf_summary: Dict[str, Optional[float]] = {}
        for cdf_sketch in self.summaries.values():
            perf_summary.update(cdf_sketch.get_summary())

        wall_s = 0.0
        if self._first_dispatch_at is not None and self._last_completion_at is not None:
            wall_s = max(0.0, self._last_completion_at - self._first_dispatch_at)
        perf_summary["chars_per_sec_aggregate"] = (
            self._total_input_chars / wall_s if wall_s > 0 else None
        )
        return perf_summary

    def finalize(self) -> EvaluationResult:
        """Finalize evaluation and return results."""
        with self.lock:
            return EvaluationResult(
                evaluator_type="audio_performance",
                channel=ChannelModality.AUDIO,
                metrics=self.get_summary(),
            )

    def get_streaming_metrics(self) -> Optional[Dict[str, Any]]:
        """Return current metrics for streaming."""
        with self.lock:
            return self.get_summary()

    def save(self, output_dir: str) -> None:
        """Save all evaluation artifacts."""
        with self.lock:
            self._save_request_level_metrics(output_dir)
            self._save_cdf_csvs(output_dir)
            self._plot_cdfs(output_dir)
            if self.channel_config.save_audio_files:
                # Save audio files to parent dir (not inside metrics/)
                parent_dir = os.path.dirname(output_dir)
                self._save_audio_files(parent_dir)

    def flush_streaming_outputs(self, output_dir: str) -> None:
        """Flush current metrics for streaming."""
        with self.lock:
            rows = self._export_request_rows(self._request_rows_streamed)
            if rows:
                self._append_request_level_rows(output_dir, rows)
                self._request_rows_streamed = len(self._completed_metrics)
            self._save_cdf_csvs(output_dir)

    # ---- Output helpers ----

    def _save_request_level_metrics(self, output_dir: str) -> None:
        """Save request-level audio metrics as JSONL."""
        path = os.path.join(output_dir, "request_level_metrics.jsonl")
        rows = self._export_request_rows(0)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _export_request_rows(self, start_index: int = 0) -> List[Dict[str, Any]]:
        rows = []
        for idx in range(start_index, len(self._completed_metrics)):
            m = self._completed_metrics[idx]
            lifecycle = self._lifecycle_timestamps[idx]
            normalized_dispatched = max(
                0.0, m.request_dispatched_at - self._request_time_reference
            )
            normalized_completed = max(
                0.0, m.client_completed_at - self._request_time_reference
            )
            rows.append(
                {
                    "request_id": m.request_id,
                    "session_id": m.session_id,
                    "session_total_requests": m.session_total_requests,
                    # Lifecycle timestamps
                    "scheduler_ready_at": lifecycle["scheduler_ready_at"],
                    "scheduler_dispatched_at": round(normalized_dispatched, 5),
                    "client_picked_up_at": lifecycle["client_picked_up_at"],
                    "client_completed_at": round(normalized_completed, 5),
                    "result_processed_at": lifecycle["result_processed_at"],
                    "ttfa": round(m.ttfa, 3),
                    "end_to_end_latency": round(m.end_to_end_latency, 3),
                    "generated_audio_duration": round(m.generated_audio_duration, 3),
                    "rtf": round(m.rtf, 5),
                    "chunk_count": m.chunk_count,
                    "pcm_byte_count": m.pcm_byte_count,
                    "input_chars": m.input_chars,
                    "input_tokens": m.input_tokens,
                }
            )
        return rows

    def _append_request_level_rows(
        self, output_dir: str, rows: List[Dict[str, Any]]
    ) -> None:
        path = os.path.join(output_dir, "request_level_metrics.jsonl")
        with open(path, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _save_cdf_csvs(self, output_dir: str) -> None:
        for metric_name, cdf_sketch in self.summaries.items():
            df = cdf_sketch._to_df()
            df.to_csv(
                os.path.join(output_dir, f"audio_{metric_name}.csv"),
                index=False,
            )

    def _plot_cdfs(self, output_dir: str) -> None:
        for metric_name, cdf_sketch in self.summaries.items():
            try:
                cdf_sketch.plot_cdf(output_dir, f"audio_{metric_name}")
            except Exception as e:
                logger.warning("Failed to plot CDF for %s: %s", metric_name, e)

    def _save_audio_files(self, output_dir: str) -> None:
        """Save collected audio buffers as WAV files."""
        if not self._audio_buffers:
            return
        audio_dir = os.path.join(output_dir, "audio_files")
        os.makedirs(audio_dir, exist_ok=True)
        for req_id, audio_data in self._audio_buffers.items():
            meta = self._audio_metadata.get(req_id, {})
            raw_pcm = meta.get("raw_pcm", False)
            sample_rate = meta.get("sample_rate", SAMPLE_RATE_DEFAULT)
            wav_path = os.path.join(audio_dir, f"request_{req_id}.wav")
            with open(wav_path, "wb") as f:
                if raw_pcm:
                    f.write(_make_wav_header(len(audio_data), sample_rate))
                f.write(audio_data)
        logger.info("Saved %d audio files to %s", len(self._audio_buffers), audio_dir)
        self._audio_buffers.clear()
        self._audio_metadata.clear()
