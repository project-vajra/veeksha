"""Performance evaluator for audio request metrics."""

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from veeksha.config.evaluator import (
    AudioChannelPerformanceConfig,
    PerformanceEvaluatorConfig,
)
from veeksha.evaluator.base import EvaluationResult
from veeksha.evaluator.cdf_sketch import CDFSketch
from veeksha.evaluator.performance.asr import (
    ASRMetricAccumulator,
    ASRRequestMetrics,
    score_asr_request,
)
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

logger = init_logger(__name__)

DEFAULT_AUDIO_SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
WAV_HEADER_BYTES = 44


def _pcm_byte_count(total_bytes: int, *, raw_pcm: bool) -> int:
    if raw_pcm:
        return total_bytes
    return max(total_bytes - WAV_HEADER_BYTES, 0)


def _audio_duration_ms(
    pcm_bytes: int, sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
) -> float:
    num_samples = pcm_bytes / BYTES_PER_SAMPLE
    return (num_samples / sample_rate) * 1000


@dataclass
class AudioRequestMetrics:
    """Metrics for a single audio request."""

    request_id: int
    session_id: int
    request_dispatched_at: float
    client_completed_at: float
    ttfc: float
    end_to_end_latency: float
    generated_audio_duration: float
    rtf: float
    chunk_count: int
    pcm_byte_count: int
    input_chars: int = 0
    input_tokens: int = 0
    audio_task: Optional[AudioTask] = None
    asr: Optional[ASRRequestMetrics] = None
    input_text: str = ""
    session_total_requests: Optional[int] = None


class AudioPerformanceEvaluator:
    """Performance evaluator for audio (TTS / STT).

    Tracks per-request TTFC, total latency, audio duration, RTF (real-time factor),
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

        # CDF sketches for aggregate metrics.
        self.summaries: Dict[str, CDFSketch] = {
            "ttfc": CDFSketch("ttfc", unit="ms"),
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
        self.asr_latency_summaries: Dict[str, CDFSketch] = {
            "time_to_first_partial": CDFSketch("time_to_first_partial", unit="ms"),
            "time_to_final_transcript": CDFSketch(
                "time_to_final_transcript", unit="ms"
            ),
        }
        self._asr_metrics = ASRMetricAccumulator()
        self._stt_request_count = 0

        # Request dispatch tracking (like text.py _pending_requests)
        self._pending_requests: Dict[int, Dict[str, Any]] = {}

        # Request-level storage for JSONL output
        self._completed_metrics: List[AudioRequestMetrics] = []
        self._lifecycle_timestamps: List[Dict[str, Optional[float]]] = []
        self._request_rows_streamed: int = 0
        self._request_time_reference: float = self.benchmark_start_time

        # Running totals for aggregate throughput
        self._total_input_chars: int = 0
        self._total_generated_audio_duration_ms: float = 0.0
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
            ttfc = cm.get("ttfc", 0.0)
            end_to_end_latency = cm.get("end_to_end_latency", 0.0)
            chunk_count = cm.get("chunk_count", 0)
            raw_pcm = bool(cm.get("raw_pcm", False))
            sample_rate = int(cm.get("sample_rate", DEFAULT_AUDIO_SAMPLE_RATE))
            # STT measures the input clip (reported via pcm_byte_count);
            # TTS / LLM_AUDIO measure generated output audio bytes.
            audio_task = cm.get("audio_task")
            if audio_task == AudioTask.STT:
                total_bytes = int(cm["pcm_byte_count"])
            elif audio_task in (AudioTask.TTS, AudioTask.LLM_AUDIO):
                audio_content = channel_response.content
                total_bytes = (
                    len(audio_content) if isinstance(audio_content, bytes) else 0
                )
            else:
                raise ValueError(
                    f"AUDIO response for request {request_id} has unknown "
                    f"audio_task={audio_task!r}; expected one of {list(AudioTask)}"
                )
            pcm_byte_count = _pcm_byte_count(total_bytes, raw_pcm=raw_pcm)
            generated_audio_duration = _audio_duration_ms(pcm_byte_count, sample_rate)
            rtf = (
                end_to_end_latency / generated_audio_duration
                if generated_audio_duration > 0
                else float("inf")
            )
            input_chars = cm.get("input_chars", 0)
            input_tokens = cm.get("input_tokens", 0)
            input_text = cm.get("input_text", "")

            session_total_requests = getattr(response, "session_total_requests", None)

            asr_metrics = None
            if audio_task == AudioTask.STT:
                self._stt_request_count += 1
                asr_metrics = score_asr_request(
                    request_id=request_id,
                    channel_metrics=cm,
                    duration_s=generated_audio_duration / 1000.0,
                    accumulator=self._asr_metrics,
                )

            metrics = AudioRequestMetrics(
                request_id=request_id,
                session_id=session_id,
                request_dispatched_at=dispatched_at,
                client_completed_at=completed_at,
                ttfc=ttfc,
                end_to_end_latency=end_to_end_latency,
                generated_audio_duration=generated_audio_duration,
                rtf=rtf,
                chunk_count=chunk_count,
                pcm_byte_count=pcm_byte_count,
                input_chars=input_chars,
                input_tokens=input_tokens,
                audio_task=audio_task,
                asr=asr_metrics,
                input_text=input_text,
                session_total_requests=session_total_requests,
            )

            self._completed_metrics.append(metrics)

            # Update aggregate throughput accumulators
            self._total_input_chars += input_chars
            self._total_generated_audio_duration_ms += generated_audio_duration
            if (
                self._first_dispatch_at is None
                or dispatched_at < self._first_dispatch_at
            ):
                self._first_dispatch_at = dispatched_at
            if (
                self._last_completion_at is None
                or completed_at > self._last_completion_at
            ):
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
            self.summaries["ttfc"].put(ttfc)
            self.summaries["end_to_end_latency"].put(end_to_end_latency)
            self.summaries["generated_audio_duration"].put(generated_audio_duration)
            self.summaries["rtf"].put(rtf)
            self.summaries["chunk_count"].put(chunk_count)
            self.summaries["input_tokens"].put(input_tokens)
            if asr_metrics is not None:
                if asr_metrics.time_to_first_partial is not None:
                    self.asr_latency_summaries["time_to_first_partial"].put(
                        asr_metrics.time_to_first_partial
                    )
                if asr_metrics.time_to_final_transcript is not None:
                    self.asr_latency_summaries["time_to_final_transcript"].put(
                        asr_metrics.time_to_final_transcript
                    )

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
        if self._stt_request_count > 0:
            for cdf_sketch in self.asr_latency_summaries.values():
                if len(cdf_sketch) > 0:
                    perf_summary.update(cdf_sketch.get_summary())
            perf_summary.update(self._asr_metrics.get_summary())

        wall_s = 0.0
        if self._first_dispatch_at is not None and self._last_completion_at is not None:
            wall_s = max(0.0, self._last_completion_at - self._first_dispatch_at)
        perf_summary["chars_per_sec_aggregate"] = (
            self._total_input_chars / wall_s if wall_s > 0 else None
        )
        perf_summary["generated_audio_seconds_per_sec_aggregate"] = (
            (self._total_generated_audio_duration_ms / 1000.0) / wall_s
            if wall_s > 0
            else None
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
            row_dict = {
                "request_id": m.request_id,
                "session_id": m.session_id,
                "session_total_requests": m.session_total_requests,
                # Lifecycle timestamps
                "scheduler_ready_at": lifecycle["scheduler_ready_at"],
                "scheduler_dispatched_at": round(normalized_dispatched, 5),
                "client_picked_up_at": lifecycle["client_picked_up_at"],
                "client_completed_at": round(normalized_completed, 5),
                "result_processed_at": lifecycle["result_processed_at"],
                "ttfc": round(m.ttfc, 3),
                "end_to_end_latency": round(m.end_to_end_latency, 3),
                "generated_audio_duration": round(m.generated_audio_duration, 3),
                "rtf": round(m.rtf, 5),
                "chunk_count": m.chunk_count,
                "pcm_byte_count": m.pcm_byte_count,
                "input_chars": m.input_chars,
                "input_tokens": m.input_tokens,
                "input_text": m.input_text,
                "audio_task": str(m.audio_task) if m.audio_task is not None else None,
            }
            if m.asr is not None:
                row_dict.update(m.asr.to_request_row())
            rows.append(row_dict)
        return rows

    def _append_request_level_rows(
        self, output_dir: str, rows: List[Dict[str, Any]]
    ) -> None:
        path = os.path.join(output_dir, "request_level_metrics.jsonl")
        with open(path, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _save_cdf_csvs(self, output_dir: str) -> None:
        summaries = dict(self.summaries)
        if self._stt_request_count > 0:
            summaries.update(
                {
                    name: sketch
                    for name, sketch in self.asr_latency_summaries.items()
                    if len(sketch) > 0
                }
            )
        for metric_name, cdf_sketch in summaries.items():
            df = cdf_sketch._to_df()
            df.to_csv(
                os.path.join(output_dir, f"audio_{metric_name}.csv"),
                index=False,
            )

    def _plot_cdfs(self, output_dir: str) -> None:
        summaries = dict(self.summaries)
        if self._stt_request_count > 0:
            summaries.update(
                {
                    name: sketch
                    for name, sketch in self.asr_latency_summaries.items()
                    if len(sketch) > 0
                }
            )
        for metric_name, cdf_sketch in summaries.items():
            try:
                cdf_sketch.plot_cdf(output_dir, f"audio_{metric_name}")
            except Exception as e:
                logger.warning("Failed to plot CDF for %s: %s", metric_name, e)
